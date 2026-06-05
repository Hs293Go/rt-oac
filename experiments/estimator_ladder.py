"""Why do first-order objectives win? Estimator-ladder test of recoverable observability.

The tension (the open question of this work): OAC/STLOG exists to recover directions observable only
at HIGH Lie order (r*>=5), yet the Phase-B sweep found a FIRST-ORDER recursive-PCRB objective wins,
and the recursive PCRB on the STLOG orbit is ~3.6 m (vs a 0.25 m batch CRLB). Two readings:
  (a) legitimate -- first-order recursive accumulation through Phi IS the same observability the STLOG
      packs into a short window via Lie derivatives, just weighted for the filter; OR
  (b) artifact -- scoring on a FIRST-ORDER EKF crippled the analysis and abandoned recoverable
      high-order / non-Gaussian (spherical-shell) observability that a better estimator would cash in.

DECISIVE TEST: run a ladder of estimators of increasing order -- EKF (1st-order linearization),
ManifoldUKF (2nd-order, unscented), and a bootstrap PARTICLE FILTER (exact posterior, represents the
non-Gaussian shell) -- CARRIED from a wrong prior (not truth-anchored) along a FIXED orbit with honest
process + measurement noise, and measure the true position error (radial vs tangential), Monte-Carlo
over noise/prior seeds. If the PF drives the TANGENTIAL error well below the EKF's, the high-order /
non-Gaussian observability is real and extractable -> reading (b); the EKF is the bottleneck. If the
PF ~ the EKF, the recursive limit is fundamental (process noise) and order is not the lever -> (a).

Feed it an orbit dump with x0 + the full input sequence us (N,8):
  * the STLOG orbit: a closed-loop dump's xrel_full[0] + u_applied (--npz <dump.npz>), or
  * a PCRB-optimized orbit: pcrb_optimize.py --dump-orbit (keys x0, us_opt/us_base) (--orbit <o.npz> --which us_opt).

    JAX_PLATFORMS=cpu uv run python experiments/estimator_ladder.py --npz /tmp/derisk/quad_estimation_planontruth.npz \\
        --particles 8000 --seeds 24
"""

import argparse
import functools

from example_lib.models import inter_quadrotor_pose as mdl
import jax
import jax.numpy as jnp
import numpy as np
from observability_aware_control import integrator

from rt_oac.error_state_ekf import ErrorStateEKF
from rt_oac.unscented_kf import ManifoldUKF

jax.config.update("jax_enable_x64", True)

INPUT_VAR = np.tile([0.05, 0.01, 0.01, 0.01], 2)
PROC_VAR = np.array([0.02, 0.02, 0.02, 1e-4, 1e-4, 1e-4, 0.05, 0.05, 0.05])
P0_DIAG = np.array([2.0, 2.0, 2.0, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0])


def renorm(x):
    x = np.array(x)
    x[3:7] /= np.linalg.norm(x[3:7])
    return x


def radial_tangential(err_vec, r):
    """Split a position error into radial (along line-of-sight) and tangential magnitudes."""
    u = r / (np.linalg.norm(r) + 1e-12)
    radial = float(np.dot(err_vec, u))
    tang = float(np.linalg.norm(err_vec - radial * u))
    return abs(radial), tang


class ParticleFilter:
    """Bootstrap PF on R^3 x SO(3) x R^3 -- the gold-standard (exact, non-Gaussian) estimator.

    Process: each particle steps the true nonlinear dynamics with its own input-noise draw and an
    additive tangent process-noise draw (matching the EKF's assumed Q). Weight: the manifold-aware
    measurement residual (range + quaternion-log) under R. Systematic resampling on low ESS. The
    posterior position estimate is the weighted particle mean -- which, unlike a Gaussian filter, can
    represent the spherical-shell ambiguity a single range leaves in the tangential subspace.
    """

    def __init__(
        self, step, observation, residual, manifold, in_cov, proc_cov, res_var, m, key
    ):
        self._step = step
        self._obs = observation
        self._residual = residual
        self._manifold = manifold
        self._in_cov_chol = jnp.linalg.cholesky(
            jnp.asarray(in_cov) + 1e-12 * jnp.eye(8)
        )
        self._proc_chol = jnp.linalg.cholesky(
            jnp.asarray(proc_cov) + 1e-12 * jnp.eye(9)
        )
        self._rinv = jnp.diag(1.0 / jnp.asarray(res_var))
        self._m = m
        self._key = key

    @functools.partial(jax.jit, static_argnames=["self"])
    def _predict(self, parts, u, dt, key):
        kin, kpr = jax.random.split(key)
        un = jax.random.normal(kin, (self._m, 8)) @ self._in_cov_chol.T
        dn = jax.random.normal(kpr, (self._m, 9)) @ self._proc_chol.T

        def one(p, ui, di):
            return self._manifold.boxplus(self._step(p, u + ui, dt), di)

        return jax.vmap(one)(parts, un, dn)

    @functools.partial(jax.jit, static_argnames=["self"])
    def _logw(self, parts, y):
        def one(p):
            nu = self._residual(p, y)
            return -0.5 * nu @ self._rinv @ nu

        return jax.vmap(one)(parts)

    def step_update(self, parts, u, y, dt):
        self._key, k = jax.random.split(self._key)
        parts = self._predict(parts, u, dt, k)
        logw = np.array(self._logw(parts, y))
        logw -= logw.max()
        w = np.exp(logw)
        w /= w.sum()
        pos_mean = np.asarray(parts)[:, 0:3].T @ w  # weighted position mean
        ess = 1.0 / np.sum(w**2)
        if ess < self._m / 2:  # systematic resampling
            self._key, k = jax.random.split(self._key)
            u0 = float(jax.random.uniform(k)) / self._m
            idx = np.searchsorted(np.cumsum(w), u0 + np.arange(self._m) / self._m)
            idx = np.clip(idx, 0, self._m - 1)
            parts = parts[idx]
        return parts, pos_mean


def run_orbit(x0, us, dt, sim, ekf, ukf, pf_factory, n_seeds, warmup):
    """Monte-Carlo the estimator ladder along the fixed orbit; return per-estimator error stats."""
    obs_var = np.concatenate([[ekf._obs_cov[0, 0]], np.full(4, ekf._obs_cov[1, 1])])
    N = us.shape[0]
    P0 = np.diag(P0_DIAG)
    L0 = np.linalg.cholesky(P0)
    acc = {k: {"rad": [], "tang": [], "pos": []} for k in ("ekf", "ukf", "pf")}
    for s in range(n_seeds):
        rng = np.random.default_rng(1000 + s)
        # truth realization (process + input noise) and measurements along the orbit
        xs_true = np.zeros((N + 1, 10))
        xs_true[0] = renorm(x0)
        ys = np.zeros((N, 5))
        for k in range(N):
            ys[k] = np.asarray(mdl.observation(jnp.asarray(xs_true[k]))) + rng.normal(
                0, np.sqrt(obs_var)
            )
            un = us[k].copy()
            un[4:] += rng.normal(
                0, np.sqrt(INPUT_VAR[4:])
            )  # input noise the filter doesn't know
            xn = np.asarray(
                sim(jnp.asarray(xs_true[k]), jnp.asarray(un))[0]
            )  # Integrator -> (x_next, aux)
            xs_true[k + 1] = renorm(
                xn
            )  # truth = deterministic dynamics + input noise (matches example)
        # common wrong prior
        x_hat0 = renorm(
            np.asarray(
                ekf._manifold.boxplus(
                    jnp.asarray(xs_true[0]), jnp.asarray(L0 @ rng.normal(size=9))
                )
            )
        )

        runs = {}
        # EKF / UKF carried
        for name, filt in (("ekf", ekf), ("ukf", ukf)):
            xh, P = x_hat0.copy(), P0.copy()
            errs = []
            for k in range(N):
                xh, P = filt.predict(
                    jnp.asarray(xh), jnp.asarray(P), jnp.asarray(us[k]), dt
                )
                xh, P = filt.update(jnp.asarray(xh), jnp.asarray(P), jnp.asarray(ys[k]))
                xh = renorm(xh)
                errs.append(xh[0:3] - xs_true[k + 1][0:3])
            runs[name] = errs
        # PF carried
        pf = pf_factory(rng.integers(1 << 30))
        parts = jax.vmap(
            lambda z: pf._manifold.boxplus(jnp.asarray(x_hat0), jnp.asarray(z))
        )((L0 @ rng.normal(size=(9, pf._m))).T)
        errs = []
        for k in range(N):
            parts, pos_mean = pf.step_update(
                parts, jnp.asarray(us[k]), jnp.asarray(ys[k]), dt
            )
            errs.append(pos_mean - xs_true[k + 1][0:3])
        runs["pf"] = errs

        for name, errs in runs.items():
            for k in range(warmup, N):
                rad, tang = radial_tangential(np.asarray(errs[k]), xs_true[k + 1][0:3])
                acc[name]["rad"].append(rad)
                acc[name]["tang"].append(tang)
                acc[name]["pos"].append(float(np.linalg.norm(errs[k])))
    out = {}
    for name, d in acc.items():
        out[name] = {
            "rms_pos": float(np.sqrt(np.mean(np.square(d["pos"])))),
            "rms_radial": float(np.sqrt(np.mean(np.square(d["rad"])))),
            "rms_tangential": float(np.sqrt(np.mean(np.square(d["tang"])))),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--npz", default=None, help="closed-loop dump (xrel_full + u_applied)"
    )
    ap.add_argument(
        "--orbit", default=None, help="pcrb_optimize --dump-orbit npz (x0 + us_*)"
    )
    ap.add_argument("--which", default="us_opt", help="for --orbit: us_opt or us_base")
    ap.add_argument("--particles", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--label", default="orbit")
    args = ap.parse_args()

    if args.npz:
        d = np.load(args.npz)
        x0, us = np.asarray(d["xrel_full"])[0], np.asarray(d["u_applied"])
    else:
        d = np.load(args.orbit)
        x0, us = np.asarray(d["x0"]), np.asarray(d[args.which])
    N = us.shape[0]
    warmup = min(20, N // 3)

    range_var, att_var = 0.01, 0.01  # matches conf noise
    res_var = np.concatenate([[range_var], np.full(3, att_var)])
    dt = 0.05
    sim = jax.jit(
        integrator.Integrator(
            mdl.dynamics, integrator.Methods.EULER, stepsize=dt, manifold=mdl.MANIFOLD
        )
    )
    common = {
        "dynamics": mdl.dynamics,
        "observation": lambda x: mdl.observation(x),
        "manifold": mdl.MANIFOLD,
        "in_cov": np.diag(INPUT_VAR),
        "obs_cov": np.diag(res_var),
        "proc_cov": np.diag(PROC_VAR),
        "method": integrator.Methods.EULER,
    }
    ekf = ErrorStateEKF(**common)
    ukf = ManifoldUKF(**common)

    def pf_factory(seed):
        return ParticleFilter(
            ekf._step,
            ekf._observation,
            ekf._residual,
            mdl.MANIFOLD,
            np.diag(INPUT_VAR),
            np.diag(PROC_VAR),
            res_var,
            args.particles,
            jax.random.PRNGKey(int(seed)),
        )

    print(
        f"orbit '{args.label}': N={N} steps, {args.seeds} seeds, PF {args.particles} particles, warmup {warmup}"
    )
    out = run_orbit(x0, us, dt, sim, ekf, ukf, pf_factory, args.seeds, warmup)
    print(f"\n{'estimator':<10}{'RMS pos':>10}{'RMS radial':>12}{'RMS tangential':>16}")
    for name in ("ekf", "ukf", "pf"):
        r = out[name]
        print(
            f"{name.upper():<10}{r['rms_pos']:>10.3f}{r['rms_radial']:>12.3f}{r['rms_tangential']:>16.3f}"
        )
    pf_t, ekf_t = out["pf"]["rms_tangential"], out["ekf"]["rms_tangential"]
    print(
        f"\nPF vs EKF tangential RMS: {ekf_t:.3f} -> {pf_t:.3f} m  (x{ekf_t / max(pf_t, 1e-9):.2f} tighter)"
    )
    print(
        "  >>1 => a better estimator recovers tangential observability the EKF leaves on the table"
    )
    print(
        "        (the high-order / non-Gaussian info is real; first-order is the bottleneck)."
    )
    print(
        "  ~1  => the recursive limit is fundamental (process noise); estimator ORDER is not the lever."
    )
    import json

    print("JSON " + json.dumps({"label": args.label, "N": N, **out}))


if __name__ == "__main__":
    main()
