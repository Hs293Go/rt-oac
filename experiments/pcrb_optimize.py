"""Phase B prototype: directly optimize the follower's orbit for recursive localizability.

Phase A (PROGRESS #16) found the recursive posterior CRLB on the follower's tangential position is
~3.6 m (3-sigma) on the STLOG-optimal orbit -- comparable to the ~1 m standoff, so the carried
controller never has a tight-enough estimate. The STLOG objective maximizes observability VOLUME,
not this recursive PCRB (#17 showed a PCRB-targeted orbit is ~2.3x tighter). This script is the
exploration tool: a self-contained differentiable optimizer over the follower's input sequence.

  decision vars  = the follower's input sequence (N x 4: thrust + 3 body rates); leader fixed;
  forward        = integrate the relative dynamics -> trajectory; accumulate BOTH
                   * the EKF Riccati at truth (F/G/H reused from ErrorStateEKF) -> recursive P, and
                   * the batch observability Gramian W_o = sum Phi^T H^T R^-1 H Phi;
  objective      = one of several forms (recursive-P trace/logdet/maxeig/posvel, or batch-W_o
                   mineig/logdet) + a soft standoff-band penalty so it can't cheat with a huge
                   parallax. The COMMON YARDSTICK reported for every objective is the recursive PCRB.

Optimized with JAX value_and_grad + scipy L-BFGS-B, multi-restart. Emits a flat metrics dict
(``--dump out.json``) and optionally the optimized orbit (``--dump-orbit out.npz``) so a sweep can
tabulate the floor, the observability<->formation Pareto, and which objective form wins.

    JAX_PLATFORMS=cpu uv run python experiments/pcrb_optimize.py \\
        --objective trace --steps 40 --band-lo 0.5 --band-hi 1.2 --iters 120 --restarts 3
"""

import argparse
import json
import pathlib

from example_lib.models import inter_quadrotor_pose as mdl
import jax
import jax.numpy as jnp
import numpy as np
from observability_aware_control import integrator
import scipy.optimize as sopt

from rt_oac.error_state_ekf import ErrorStateEKF
from rt_oac.scenario import build_scenario

jax.config.update("jax_enable_x64", True)

# objective forms: name -> reducer(P_pos_blocks, P_vel_blocks, W_o_final) -> scalar to MINIMIZE.
# P-based forms act on the steady-state (second-half) recursive covariance; W_o forms on the final
# accumulated batch Gramian (a process-noise-free "pure observability" surrogate).


def _logdet(M):
    return jnp.sum(jnp.log(jnp.linalg.eigvalsh(M) + 1e-12))


def _mineig(M):
    return jnp.linalg.eigvalsh(M)[0]


OBJECTIVES = {
    "trace": lambda pp, pv, w: jnp.mean(jnp.trace(pp, axis1=1, axis2=2)),
    "logdet": lambda pp, pv, w: jnp.mean(jax.vmap(_logdet)(pp)),
    "maxeig": lambda pp, pv, w: jnp.mean(
        jax.vmap(lambda m: jnp.linalg.eigvalsh(m)[-1])(pp)
    ),
    "posvel": lambda pp, pv, w: jnp.mean(
        jnp.trace(pp, axis1=1, axis2=2) + jnp.trace(pv, axis1=1, axis2=2)
    ),
    "wo_mineig": lambda pp, pv, w: -_mineig(
        w
    ),  # maximize worst accumulated info direction
    "wo_logdet": lambda pp, pv, w: -_logdet(w),  # maximize accumulated info volume
}


def build_fns(
    ekf, dt, in_cov, proc_cov, R4, rinv, x0, u_leader, P0, band, w_band, objective
):
    """Return jitted (cost_and_grad, evaluate) over an N-step rollout from x0."""
    n = ekf._tangent_dim
    zero = jnp.zeros(n)
    lo2, hi2 = band[0] ** 2, band[1] ** 2
    obj_fn = OBJECTIVES[objective]

    def step_jacobians(x, u):
        x_next = ekf._step(x, u, dt)

        def err(d):
            return ekf._boxminus(ekf._step(ekf._manifold.boxplus(x, d), u, dt), x_next)

        def imap(ui):
            return ekf._boxminus(ekf._step(x, ui, dt), x_next)

        return x_next, jax.jacobian(err)(zero), jax.jacobian(imap)(u)

    def meas_jac(x):
        y = ekf._observation(x)
        return -jax.jacobian(lambda d: ekf._residual(ekf._manifold.boxplus(x, d), y))(
            zero
        )

    def rollout(u_foll):
        us = jnp.concatenate([u_leader, u_foll], axis=1)  # (N, 8)

        def scan_step(carry, u):
            x, P, phi, W = carry
            x_next, F, G = step_jacobians(x, u)
            P = F @ P @ F.T + G @ in_cov @ G.T + proc_cov  # predict at truth
            H = meas_jac(x_next)
            S = H @ P @ H.T + R4
            P -= P @ H.T @ jnp.linalg.solve(S, H @ P)  # update at truth
            P = 0.5 * (P + P.T)
            phi = F @ phi
            Hphi = H @ phi
            W += Hphi.T @ rinv @ Hphi  # batch observability Gramian (x0)
            return (x_next, P, phi, W), (P[0:3, 0:3], P[6:9, 6:9], x_next[0:3])

        eye = jnp.eye(n)
        (_, _, _, W), (pp, pv, pos) = jax.lax.scan(
            scan_step, (x0, P0, eye, jnp.zeros((n, n))), us
        )
        return pp, pv, pos, W

    def cost(u_foll):
        pp, pv, pos, W = rollout(u_foll)
        half = pp.shape[0] // 2
        j = obj_fn(pp[half:], pv[half:], W)
        d2 = jnp.sum(pos**2, axis=1)
        viol = jnp.maximum(0.0, d2 - hi2) + jnp.maximum(0.0, lo2 - d2)
        return j + w_band * jnp.mean(viol**2)

    @jax.jit
    def cost_and_grad(u_foll):
        return jax.value_and_grad(cost)(u_foll)

    @jax.jit
    def evaluate(u_foll):
        """Common-yardstick scalars (recursive PCRB etc.) + the orbit positions."""
        pp, pv, pos, W = rollout(u_foll)
        half = pp.shape[0] // 2
        sig_pos = 3.0 * jnp.sqrt(jnp.mean(jnp.trace(pp[half:], axis1=1, axis2=2)) / 3.0)
        sig_vel = 3.0 * jnp.sqrt(jnp.mean(jnp.trace(pv[half:], axis1=1, axis2=2)) / 3.0)
        d = jnp.linalg.norm(pos, axis=1)
        viol = jnp.maximum(0.0, d - band[1]) + jnp.maximum(0.0, band[0] - d)
        evals = jnp.linalg.eigvalsh(W)
        scalars = jnp.array([
            sig_pos,
            sig_vel,
            jnp.mean(viol**2),
            evals[0],
            evals[-1] / (evals[0] + 1e-12),
            d.min(),
            d.max(),
            d.mean(),
        ])
        return scalars, pos

    return cost_and_grad, evaluate


def los_rotation(pos):
    """Total swept angle of the unit line-of-sight over the trajectory (radians)."""
    u = pos / (np.linalg.norm(pos, axis=1, keepdims=True) + 1e-12)
    dots = np.clip(np.sum(u[1:] * u[:-1], axis=1), -1.0, 1.0)
    return float(np.sum(np.arccos(dots)))


def optimize(args):
    sc = build_scenario()
    dt = float(sc.cfg["sim"]["integrator_dt"])
    band = (args.band_lo, args.band_hi)
    range_var = float(sc.cfg["noise"]["range_var"])
    att_var = float(sc.cfg["noise"]["att_var"])
    res_var = np.concatenate([[range_var], np.full(3, att_var)])
    in_cov = jnp.diag(jnp.asarray(np.tile([0.05, 0.01, 0.01, 0.01], 2)))
    proc_cov = jnp.diag(
        jnp.asarray([0.02, 0.02, 0.02, 1e-4, 1e-4, 1e-4, 0.05, 0.05, 0.05])
    )
    R4 = jnp.diag(jnp.asarray(res_var))
    rinv = jnp.diag(1.0 / jnp.asarray(res_var))
    P0 = (
        jnp.diag(jnp.asarray([2.0, 2.0, 2.0, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0]))
        * args.p0_scale
    )

    ekf = ErrorStateEKF(
        mdl.dynamics,
        lambda x: mdl.observation(x),
        mdl.MANIFOLD,
        in_cov=np.eye(8),
        obs_cov=np.diag(res_var),
        method=integrator.Methods.EULER,
    )
    x0 = jnp.asarray(np.concatenate([[1.0, 0.0, 0.0], _quat_z(20), [0.0, 1.0, 0.0]]))
    guess = np.asarray(sc.reference_guess(0))
    if args.steps > guess.shape[0]:
        pad = np.repeat(guess[-1:], args.steps - guess.shape[0], axis=0)
        guess = np.concatenate([guess, pad], axis=0)
    N = min(args.steps, guess.shape[0])
    guess = guess[:N]
    u_leader = np.array(guess[:, 0:4])
    if (
        args.leader == "maneuver"
    ):  # gentle leader body-rate weave (tests leader-input effect)
        u_leader[:, 1] = 0.3 * np.sin(np.arange(N) * 0.4)
        u_leader[:, 2] = 0.3 * np.cos(np.arange(N) * 0.4)
    u_leader = jnp.asarray(u_leader)
    u_foll0 = np.array(guess[:, 4:8])

    cost_and_grad, evaluate = build_fns(
        ekf,
        dt,
        in_cov,
        proc_cov,
        R4,
        rinv,
        x0,
        u_leader,
        P0,
        band,
        args.wband,
        args.objective,
    )

    def scipy_obj(flat):
        v, g = cost_and_grad(jnp.asarray(flat.reshape(N, 4)))
        return float(v), np.asarray(g, dtype=np.float64).ravel()

    rng = np.random.default_rng(args.seed)
    best, best_u = None, None
    for r in range(args.restarts):
        u_init = u_foll0 + (
            0.0 if r == 0 else rng.normal(0, args.restart_jitter, u_foll0.shape)
        )
        res = sopt.minimize(
            scipy_obj,
            u_init.ravel(),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": args.iters},
        )
        if best is None or res.fun < best:
            best, best_u = res.fun, res.x.reshape(N, 4)

    # baseline (warm-start orbit) and optimized metrics on the common yardstick
    base_s, pos_base = evaluate(jnp.asarray(u_foll0))
    opt_s, pos_opt = evaluate(jnp.asarray(best_u))
    base_s, opt_s = np.asarray(base_s), np.asarray(opt_s)
    pos_base, pos_opt = np.asarray(pos_base), np.asarray(pos_opt)
    # full input sequences (leader+follower) and x0 -- replayable through the estimator ladder
    us_opt = np.concatenate([np.asarray(u_leader), np.asarray(best_u)], axis=1)
    us_base = np.concatenate([np.asarray(u_leader), np.asarray(u_foll0)], axis=1)
    orbit_arrays = {
        "pos_opt": pos_opt,
        "pos_base": pos_base,
        "us_opt": us_opt,
        "us_base": us_base,
        "x0": np.asarray(x0),
    }
    return (
        {
            "objective": args.objective,
            "steps": N,
            "band": list(band),
            "wband": args.wband,
            "leader": args.leader,
            "p0_scale": args.p0_scale,
            "restarts": args.restarts,
            "base_sig_pos": float(base_s[0]),
            "base_sig_vel": float(base_s[1]),
            "base_los_rotation": los_rotation(pos_base),
            "opt_sig_pos": float(opt_s[0]),
            "opt_sig_vel": float(opt_s[1]),
            "opt_band_viol": float(opt_s[2]),
            "opt_wo_mineig": float(opt_s[3]),
            "opt_wo_cond": float(opt_s[4]),
            "opt_dist_min": float(opt_s[5]),
            "opt_dist_max": float(opt_s[6]),
            "opt_dist_mean": float(opt_s[7]),
            "opt_los_rotation": los_rotation(pos_opt),
            "tighter_x": float(base_s[0]) / max(float(opt_s[0]), 1e-9),
        },
        orbit_arrays,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="trace", choices=list(OBJECTIVES))
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--band-lo", type=float, default=0.5, dest="band_lo")
    ap.add_argument("--band-hi", type=float, default=1.2, dest="band_hi")
    ap.add_argument("--wband", type=float, default=50.0)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--restart-jitter", type=float, default=0.05, dest="restart_jitter")
    ap.add_argument("--leader", default="hover", choices=["hover", "maneuver"])
    ap.add_argument("--p0-scale", type=float, default=1.0, dest="p0_scale")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump", default=None, help="write the metrics dict as JSON here")
    ap.add_argument(
        "--dump-orbit", default=None, dest="dump_orbit", help="npz of opt/base orbits"
    )
    args = ap.parse_args()

    metrics, orbit = optimize(args)
    print(json.dumps(metrics, indent=2))
    print(
        f"\n[{args.objective} | band {metrics['band']} | N={metrics['steps']} | {args.leader}] "
        f"PCRB 3sig_pos {metrics['base_sig_pos']:.2f} -> {metrics['opt_sig_pos']:.2f} m "
        f"(x{metrics['tighter_x']:.2f}); dist [{metrics['opt_dist_min']:.2f},{metrics['opt_dist_max']:.2f}] "
        f"viol {metrics['opt_band_viol']:.1e}; LOS rot {metrics['base_los_rotation']:.2f}->"
        f"{metrics['opt_los_rotation']:.2f} rad; vel 3sig {metrics['opt_sig_vel']:.2f}"
    )
    if args.dump:
        pathlib.Path(args.dump).write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
    if args.dump_orbit:
        np.savez(args.dump_orbit, **orbit)


def _quat_z(deg):
    a = np.deg2rad(deg) / 2
    return np.array([0.0, 0.0, np.sin(a), np.cos(a)])


if __name__ == "__main__":
    main()
