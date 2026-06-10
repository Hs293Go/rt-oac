r"""DECISIVE EXPERIMENT: belief-space (dual-control) OAC on the carried-O1 quad corner.

The question that reframes the whole program (memo `docs/ho_observability_directions.md`, next-step #1):
the carried full-quad O1 loop DIVERGES under the deterministic "maximize the STLOG" objective (o1_corner.py:
full ESEKF ~12% bounded; the O1-on-flat variant is NON-MONOTONIC in solver effort -- optimizing observability
HARDER destabilizes the carried estimate). Is that divergence ESTIMATOR-LIMITED (the planning representation is
wrong -- it maximizes from-scratch observability and over-excites) or INFORMATION-LIMITED (a genuine dead end)?

Belief-space planning tests it. Replace "maximize STLOG" with "minimize the predicted POSTERIOR covariance":
seed the planner with the CARRIED filter covariance P0, propagate the ESEKF Riccati (predict F P F' + GQG' + Qt;
update Joseph) along the planned trajectory using the FILTER'S OWN linearization (so the planned belief matches
the real filter exactly), and minimize log-det of the terminal position+velocity covariance. This is
self-limiting by construction: a dt-suppressed HO direction yields ~no covariance reduction so the optimizer
won't pay control effort for it, an already-known direction (small P0) gives diminishing returns, and process
noise Q caps the net gain -- the optimizer cannot manufacture free observability the filter can't keep.

EVERYTHING else (model, ESEKF, noise, horizon, constraint, truth sim) is held identical to o1_corner's diverging
deterministic runs; only the OBJECTIVE differs. The smoking-gun signal is MONOTONICITY: belief-space should get
BETTER with solver effort where the deterministic O1 got worse.

GO  (estimator-limited, rescuable): NEES stays in the chi^2 band AND %bounded beats deterministic O1 toward O0,
    WITH monotone (or non-inverting) error-vs-solver-effort.
NO-GO (information-limited / consistency-limited): collapses to trivial low-excitation (no gain over dead
    reckoning) OR drives log-det P down while the TRUE error / NEES blow up (the filter's self-report is
    confident-wrong -- the lever is a better FILTER, not better planning).

    PYTHONPATH=src JAX_PLATFORMS=cpu uv run python experiments/belief_space_oac.py --seeds 20
"""

import argparse
from collections import namedtuple

from example_lib.models import inter_quadrotor_pose as mdl
import jax
import jax.numpy as jnp
import numpy as np
import o1_corner as o1c
from observability_aware_control import integrator, observability_cost

from rt_oac import metrics
from rt_oac.controller import RTController

IX = (
    0,
    1,
    2,
    6,
    7,
    8,
)  # position + velocity TANGENT indices (att is 3,4,5) -- the O1 observability target
_ObjResult = namedtuple("_ObjResult", ["objective"])


class BeliefSpaceCost:
    """ObservabilityCost-compatible belief-space cost: minimize log-det of the predicted POSTERIOR covariance
    over the horizon (position+velocity sub-block), seeded with the carried P0 threaded via ``p_ref``. The
    covariance is propagated with the ESEKF's OWN F/G/H (the filter's linearization), so the planner optimizes
    the covariance the filter will actually carry -- not a from-scratch observability Gramian."""

    def __init__(self, ekf, *, stlog_dt=o1c.STLOG_DT, default_p0=None):
        self._ekf = ekf
        self._dt = float(stlog_dt)
        self._obs = ekf._observation
        self._tan = ekf._tangent_dim
        self._default_p0 = (
            np.eye(self._tan) if default_p0 is None else np.asarray(default_p0)
        )
        # delegate the trajectory rollout (for the interrobot-distance constraint) to a plain ObservabilityCost
        # on the same model -> the hard distance constraint behaves identically to the deterministic O1 planner
        self._roll = observability_cost.ObservabilityCost(
            mdl.dynamics,
            o1c.obs2,
            o1c.DT,
            gramian_kw={"order": 1, "var": o1c.VAR2, "manifold": mdl.MANIFOLD},
            gramian_metric=metrics.neg_softmin_eig,
            observed_indices=(),
        )

    def eval_integrator(self, x0, us):
        return self._roll.eval_integrator(x0, us)

    def _belief_logdet(self, x0, us, p0):
        # lax.scan (one compiled step body, NOT 20 unrolled copies -> tractable compile) propagating the
        # filter's covariance; each step re-derives F/G/H by autodiff of the ESEKF's own manifold maps.
        ekf, dt, tan = self._ekf, self._dt, self._tan
        zero = jnp.zeros(tan)
        ident = jnp.eye(tan)

        def step(carry, u):
            x, P = carry
            x_next = ekf._step(x, u, dt)
            # predict: F, G from the filter's manifold step linearization
            F = jax.jacobian(
                lambda d: ekf._boxminus(
                    ekf._step(ekf._manifold.boxplus(x, d), u, dt), x_next
                )
            )(zero)
            G = jax.jacobian(lambda uu: ekf._boxminus(ekf._step(x, uu, dt), x_next))(u)
            P = F @ P @ F.T + G @ ekf._in_cov @ G.T
            if ekf._proc_cov is not None:
                P = P + ekf._proc_cov  # noqa: PLR6104 (jax arrays: explicit rebind, not in-place)
            P = 0.5 * (P + P.T)
            # update: H from the filter's residual linearization (covariance is measurement-independent)
            y0 = ekf._observation(
                x_next
            )  # predicted measurement -> nu = 0; H independent of the value of y
            H = -jax.jacobian(
                lambda d: ekf._residual(ekf._manifold.boxplus(x_next, d), y0)
            )(zero)
            S = H @ P @ H.T + ekf._obs_cov
            K = jnp.linalg.solve(S.T, H @ P).T
            ikh = ident - K @ H
            P = ikh @ P @ ikh.T + K @ ekf._obs_cov @ K.T
            P = 0.5 * (P + P.T)
            return (x_next, P), None

        (_, p_final), _ = jax.lax.scan(step, (x0, p0), us)
        sub = p_final[jnp.ix_(jnp.asarray(IX), jnp.asarray(IX))]
        _, logdet = jnp.linalg.slogdet(sub + 1e-12 * jnp.eye(len(IX)))
        return (
            logdet  # MINIMIZE -> shrink the posterior covariance the filter will carry
        )

    def evaluate(self, x0, us, dt, p_ref, weight):
        p0 = jnp.reshape(p_ref, (self._tan, self._tan))
        return _ObjResult(objective=self._belief_logdet(x0, us, p0))

    def __call__(
        self, x0, us, dt
    ):  # fallback (p_ref=None); the loop always threads the carried P0
        return _ObjResult(
            objective=self._belief_logdet(x0, us, jnp.asarray(self._default_p0))
        )


def build_belief(lean, maxiter, *, proc_scale=1.0, rate_scale=1.0):
    """Build (ekf, ctrl): a carried ESEKF + a belief-space RTController whose cost reuses that ekf's
    linearization. Same lb/ub/constraint/solver as o1_corner.build_o1_oac -- only the objective differs."""
    in_cov = np.diag(
        np.r_[1e-4, np.full(3, 1e-4), o1c.ACCEL_NOISE**2, np.full(3, o1c.GYRO_NOISE**2)]
    )
    if lean:
        ekf = o1c.LeanESEKF(
            mdl.dynamics,
            o1c.obs_lean,
            mdl.MANIFOLD,
            in_cov=in_cov,
            obs_cov=np.diag(
                np.r_[
                    o1c.RANGE_VAR,
                    o1c.RANGE_VAR,
                    np.full(3, o1c.ATT_VAR),
                    np.full(3, o1c.VEL_VAR),
                ]
            ),
            proc_cov=proc_scale * o1c.PROC_FLOOR,
        )
    else:
        ekf = o1c.TwoRangeESEKF(
            mdl.dynamics,
            o1c.obs2,
            mdl.MANIFOLD,
            in_cov=in_cov,
            obs_cov=np.diag(
                np.r_[o1c.RANGE_VAR, o1c.RANGE_VAR, np.full(3, o1c.ATT_VAR)]
            ),
            proc_cov=proc_scale * o1c.PROC_FLOOR,
        )
    p0_default = np.diag(np.r_[np.full(3, 4.0), np.full(3, 0.01), np.full(3, 0.25)])
    lb = o1c.LB * np.r_[1.0, rate_scale, rate_scale, rate_scale]
    ub = o1c.UB * np.r_[1.0, rate_scale, rate_scale, rate_scale]
    ctrl = RTController(
        BeliefSpaceCost(ekf, default_p0=p0_default),
        stlog_dt=o1c.STLOG_DT,
        lb=lb,
        ub=ub,
        n_inputs=8,
        follower_indices=(4, 5, 6, 7),
        method="SLSQP",
        maxiter=maxiter,
        constraint=mdl.interrobot_distance_squared,
        constraint_bounds=o1c.DIST2,
    )
    return ekf, ctrl


def closed_loop_belief(ekf, ctrl, seed, *, lean=False):
    """o1_corner.closed_loop, but the OA is the belief-space controller and the carried covariance P is
    THREADED into each solve (p_ref=P.ravel()). Everything else -- truth quad+drag, IMU predict, 2-range
    (+vel/att) update, metrics -- is identical, so only the objective changes."""
    rng = np.random.default_rng(seed)
    qsim = jax.jit(
        integrator.Integrator(o1c._quad_drag, integrator.Methods.RK4, stepsize=o1c.DT)
    )
    x_l1 = np.r_[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    x_f = np.r_[0.0, 2.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    x_true = np.array(mdl.from_absolute_state(jnp.asarray(x_l1), jnp.asarray(x_f)))
    rad = x_true[0:3] / np.linalg.norm(x_true[0:3])
    that = np.cross(rad, [0.0, 0.0, 1.0])
    that /= np.linalg.norm(that)
    off = np.r_[2.0 * that, np.zeros(6)]
    x_hat = np.array(mdl.MANIFOLD.boxplus(jnp.asarray(x_true), jnp.asarray(off)))
    P = np.diag(np.r_[np.full(3, 4.0), np.full(3, 0.01), np.full(3, 0.25)])
    b_a = rng.normal(0, o1c.ACCEL_BIAS_STD)
    prev_u = None
    errs, nees, dists, logdets = [], [], [], []
    for _i in range(o1c.STEPS):
        guess = np.tile(np.r_[o1c.U_HOVER, o1c.U_HOVER], (o1c.WINDOW, 1))
        if prev_u is not None:
            guess[:, 4:8] = np.vstack([prev_u[1:, 4:8], prev_u[-1:, 4:8]])
        res = ctrl.solve(jnp.asarray(x_hat), jnp.asarray(guess), p_ref=P.ravel())
        prev_u = res.u
        logdets.append(float(res.objective))
        f_f, w_f = float(res.u[0, 4]), np.asarray(res.u[0, 5:8])
        x_f = np.array(qsim(jnp.asarray(x_f), jnp.asarray(np.r_[f_f, w_f]))[0])
        x_f[3:7] /= np.linalg.norm(x_f[3:7])
        drag_body = np.array(
            o1c.elmath.quaternion_rotate_point(
                o1c.elmath.quaternion_inverse(jnp.asarray(x_f[3:7])),
                jnp.asarray(-o1c.K_DRAG * x_f[7:10]),
            )
        )
        f_imu = f_f + drag_body[2] + b_a + rng.normal(0, o1c.ACCEL_NOISE)
        w_imu = w_f + rng.normal(0, o1c.GYRO_NOISE, 3)
        u_imu = np.r_[o1c.U_HOVER, f_imu, w_imu]
        x_hat, P = ekf.predict(
            jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(u_imu), o1c.DT
        )
        x_true = np.array(mdl.from_absolute_state(jnp.asarray(x_l1), jnp.asarray(x_f)))
        obs_fn = o1c.obs_lean if lean else o1c.obs2
        y = np.array(obs_fn(jnp.asarray(x_true)))
        y[0:2] += rng.normal(0, np.sqrt(o1c.RANGE_VAR), 2)
        y[2:6] = o1c._att_noise(x_true[3:7], rng)
        if lean:
            y[6:9] = x_true[7:10] + rng.normal(0, np.sqrt(o1c.VEL_VAR), 3)
        x_hat, P = ekf.update(jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(y))
        x_hat, P = np.array(x_hat), np.array(P)
        x_hat[3:7] /= np.linalg.norm(x_hat[3:7])
        p_f_hat = np.array(
            mdl.to_absolute_state(jnp.asarray(x_l1), jnp.asarray(x_hat))
        )[10:13]
        e = np.array(ekf._boxminus(jnp.asarray(x_hat), jnp.asarray(x_true)))
        errs.append(float(np.linalg.norm(p_f_hat - x_f[0:3])))
        nees.append(
            float(e[0:3] @ np.linalg.solve(P[0:3, 0:3] + 1e-9 * np.eye(3), e[0:3]))
        )
        dists.append(float(np.linalg.norm(x_true[0:3])))
    return np.array(errs), np.array(nees), np.array(dists), np.array(logdets)


def _summary(rows):
    r = np.array(rows)
    return (
        np.median(r[:, 0]),
        np.percentile(r[:, 0], 90),
        np.median(r[:, 1]),
        100 * np.mean(r[:, 2]),
        np.median(r[:, 3]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument(
        "--lean",
        action="store_true",
        help="lean estimator (vel+att measured) instead of full",
    )
    ap.add_argument("--maxiters", type=int, nargs="+", default=[6, 15])
    ap.add_argument(
        "--det-too", action="store_true", help="also run the deterministic-O1 baseline"
    )
    args = ap.parse_args()
    est = (
        "LEAN (vel+att measured)"
        if args.lean
        else "FULL range-only ESEKF (the diverging corner)"
    )
    print(
        "=== DECISIVE: belief-space (dual-control) OAC on the carried-O1 quad corner ==="
    )
    print(
        f"{o1c.STEPS} steps, {args.seeds} seeds, {est}; tangential init 2.0 m; "
        "bounded = recovery<1.0 m AND formation in [0.5,6] m."
    )
    print(
        "MONOTONICITY is the signal: belief-space should improve (not invert) with solver effort.\n"
    )
    print(
        f"{'config':>30} {'rec_med':>8} {'rec_p90':>8} {'NEES':>7} {'%bnd':>6} {'distMed':>8}"
    )

    def bounded(errs, dists):
        return errs[-1] < 1.0 and np.min(dists) > 0.5 and np.max(dists) < 6.0

    if args.det_too:
        for mi in args.maxiters:
            ctrl, _ = o1c.build_o1_oac(two_range=True, maxiter=mi)
            rows = []
            for s in range(args.seeds):
                e, n, d = o1c.closed_loop(ctrl, s, driven="imu", lean=args.lean)
                rows.append((e[-1], np.median(n), bounded(e, d), np.median(d)))
            rm, rp, nm, pb, dm = _summary(rows)
            print(
                f"{'O1 deterministic STLOG m' + str(mi):>30} {rm:>8.2f} {rp:>8.2f} {nm:>7.1f} {pb:>5.0f}% {dm:>8.2f}"
            )

    for mi in args.maxiters:
        ekf, ctrl = build_belief(args.lean, mi)
        rows = []
        for s in range(args.seeds):
            e, n, d, _ld = closed_loop_belief(ekf, ctrl, s, lean=args.lean)
            rows.append((e[-1], np.median(n), bounded(e, d), np.median(d)))
        rm, rp, nm, pb, dm = _summary(rows)
        print(
            f"{'belief-space logdet-P m' + str(mi):>30} {rm:>8.2f} {rp:>8.2f} {nm:>7.1f} {pb:>5.0f}% {dm:>8.2f}"
        )

    print(
        "\nVERDICT: GO (estimator-limited, rescuable) if belief-space beats deterministic O1's ~12% toward "
        "O0's 95%, NEES in band, AND improves monotonically with solver effort (the deterministic O1 "
        "inverts). NO-GO (information/consistency-limited) if it collapses to trivial excitation or drives "
        "log-det P down while NEES blows up (confident-wrong -> the lever is a better filter, not planning)."
    )


if __name__ == "__main__":
    main()
