r"""Ladder rung 2: 3D point-mass range-only cooperative localization -- does beneficial OAC survive the
dimensionality jump (planar's 1 unobservable tangential -> 3D's 2 tangential directions)?

Minimal single-variable step up from the validated planar rung (rung 1, PROGRESS §9): a 3D single-integrator point
robot pair. State [leader_pos(3), follower_pos(3)]; input [leader_vel(3), follower_vel(3)]; dynamics
xdot = u. Observation = [leader_pos(3), inter-robot range] -- the leader is measured (GPS), the
follower 3D position is observable ONLY via range, so its 2 directions PERPENDICULAR to the range
(tangential-horizontal + tangential-vertical) are motion-observable only. The radial is range-observed.

Sweeps seed x initial-error-direction x mode, mirroring experiments/planar_oac_validate.py. Prediction
(carried over from planar): OAC beats no-OAC most on the two TANGENTIAL directions, least on the radial;
a dumb weave fails; oac_tight (fixed range) keeps the benefit.

    PYTHONPATH=src JAX_PLATFORMS=cpu uv run python experiments/point3d_oac_validate.py --seeds 20
"""

import argparse

from example_lib.misc import simple_ekf
import jax
import jax.numpy as jnp
import numpy as np
from observability_aware_control import integrator, observability_cost

from rt_oac import metrics, tracking_cost
from rt_oac.balanced_cost import make_balanced
from rt_oac.controller import RTController

DT, STLOG_DT, WINDOW, STEPS = 0.1, 0.2, 5, 120
ORDER = (
    2  # Lie order: single-integrator range needs >=2 to see the tangential (curvature)
)
VAR = np.r_[np.full(3, 1e-1), 1e-2]  # [leader pos (3), range]
INPUT_VAR = np.r_[
    np.full(3, 1e-2), np.full(3, 1e-2)
]  # process noise on [leader vel, follower vel]
X0 = np.array([
    0.0,
    0.0,
    0.0,
    0.0,
    5.0,
    0.0,
])  # leader at origin, follower 5 m away in +y
LEADER_U = np.array([1.0, 0.0, 0.0])  # leader cruises +x
DIST_BOUNDS = (0.2, 7.0)
P0 = np.diag([
    1e-3,
    1e-3,
    1e-3,
    2.0,
    2.0,
    2.0,
])  # large uncertainty on follower 3D position
FB_LB, FB_UB = (
    np.array([-1.0, -2.0, -2.0]),
    np.array([3.0, 2.0, 2.0]),
)  # follower velocity bounds
HYBRID_RHO, HYBRID_WEIGHT = 1.0, 1.0


# --- 3D single-integrator point-pair model (defined inline) ---
def dynamics(x, u):
    return jnp.asarray(u)  # xdot = u  (leader & follower velocities)


def observation(x, u=None):
    lp, fp = x[0:3], x[3:6]
    rng = jnp.linalg.norm(fp - lp)
    return jnp.concatenate([lp, jnp.array([rng])])


def interrobot_distance(x0, us, cost):
    xs, _ = cost.eval_integrator(x0, us)
    return jnp.linalg.norm(xs[:, 3:6] - xs[:, 0:3], axis=1)


# range direction (leader->follower) and an orthonormal tangential basis (2 directions in 3D)
_r = X0[3:6] - X0[0:3]
RHAT = _r / np.linalg.norm(_r)  # radial (range-observed)
_tmp = np.array([1.0, 0.0, 0.0])  # not parallel to RHAT (=+y)
THAT_H = np.cross(RHAT, np.array([0.0, 0.0, 1.0]))  # horizontal tangential
THAT_H /= np.linalg.norm(THAT_H)
THAT_V = np.cross(RHAT, THAT_H)  # the other tangential (has vertical comp)
THAT_V /= np.linalg.norm(THAT_V)


METRIC = (
    metrics.neg_logdet
)  # swappable: log-det (volume, can drive-away) vs softmin-eig (worst dir)


def _cost():
    return observability_cost.ObservabilityCost(
        dynamics,
        observation,
        DT,
        gramian_kw={"order": ORDER, "var": VAR},
        gramian_metric=METRIC,
        observed_indices=(),
    )


def build_oac(dist_bounds=DIST_BOUNDS):
    return RTController(
        _cost(),
        stlog_dt=STLOG_DT,
        lb=FB_LB,
        ub=FB_UB,
        n_inputs=6,
        follower_indices=(3, 4, 5),
        method="SLSQP",
        maxiter=6,
        constraint=interrobot_distance,
        constraint_bounds=dist_bounds,
        constraint_mode="hard",
    )


def build_hybrid(rho=HYBRID_RHO):
    # rho is the tracking NORMALIZER (s_track = rho^2 * window divides j_track): LOW rho = tracking-
    # dominant (over-damps observability), HIGH rho = observability-dominant + only a LIGHT formation
    # anchor (enough to prevent the rare drive-away runaway while preserving the OAC benefit).
    cost = _cost()
    ref_us = np.tile(np.r_[LEADER_U, 1.0, 0.0, 0.0], (WINDOW, 1))
    s_obs = abs(float(cost(jnp.asarray(X0), jnp.asarray(ref_us), STLOG_DT).objective))
    track = tracking_cost.quadratic_tracking_cost(position_indices=(3, 4, 5), w_pos=1.0)
    bal = make_balanced(
        cost, track, scheme="normalized", s_track=rho**2 * WINDOW, s_obs=s_obs
    )
    return RTController(
        bal,
        stlog_dt=STLOG_DT,
        lb=FB_LB,
        ub=FB_UB,
        n_inputs=6,
        follower_indices=(3, 4, 5),
        method="SLSQP",
        maxiter=6,
        constraint=interrobot_distance,
        constraint_bounds=DIST_BOUNDS,
        constraint_mode="hard",
    )


FORMATION_OFFSET = X0[3:6] - X0[0:3]


def leader_pred(leader_pos):
    ks = np.arange(WINDOW)[:, None]
    return leader_pos[None, :] + ks * DT * LEADER_U[None, :]


def run_once(mode, ctrl, seed, offset):
    rng = np.random.default_rng(seed)
    sim = jax.jit(integrator.Integrator(dynamics, integrator.Methods.RK4, stepsize=DT))
    ekf = simple_ekf.SimpleEKF(
        lambda x, u, dt: x + dt * dynamics(x, u),
        observation,
        in_cov=np.diag(INPUT_VAR),
        obs_cov=np.diag(VAR),
    )
    x_true = X0.copy()
    P = P0.copy()
    x_hat = x_true + np.r_[0, 0, 0, offset]
    prev_u, errs, neess, dists = None, [], [], []
    for i in range(STEPS):
        if mode == "noac":
            u_foll = LEADER_U.copy()  # match the leader (drive straight alongside)
        elif mode == "weave":
            u_foll = LEADER_U + np.array([
                0.0,
                np.sin(2 * np.pi * i / 15),
                np.cos(2 * np.pi * i / 15),
            ])
        else:
            guess = np.tile(np.r_[LEADER_U, LEADER_U], (WINDOW, 1))
            if prev_u is not None:
                guess[:, 3:6] = np.vstack([prev_u[1:, 3:6], prev_u[-1:, 3:6]])
            else:
                # the straight-line guess is a SYMMETRIC critical point of log-det (zero gradient,
                # SLSQP can't move); perturb the first guess to break symmetry and escape it. After
                # step 0 the warm-start (maneuvering solution) keeps the guess off the critical point.
                kk = np.arange(WINDOW)
                guess[:, 3:6] += 0.3 * np.c_[np.sin(kk), np.cos(kk), np.sin(kk + 2.0)]
            if mode == "hybrid":
                p_ref = leader_pred(x_hat[0:3]) + FORMATION_OFFSET
                res = ctrl.solve(x_hat, guess, p_ref=p_ref, weight=HYBRID_WEIGHT)
            else:
                res = ctrl.solve(x_hat, guess)
            prev_u = res.u
            u_foll = np.asarray(res.u[0, 3:6])
        u = np.r_[LEADER_U, u_foll]
        u_applied = u + np.r_[np.zeros(3), rng.normal(0, np.sqrt(INPUT_VAR[3:]))]
        x_hat, P = ekf.predict(
            jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(u_applied), DT
        )
        x_true = np.array(sim(jnp.asarray(x_true), jnp.asarray(u_applied))[0])
        y = np.array(observation(jnp.asarray(x_true))) + rng.normal(0, np.sqrt(VAR))
        x_hat, P = ekf.update(jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(y))
        x_hat, P = np.array(x_hat), np.array(P)
        ef = x_hat[3:6] - x_true[3:6]
        errs.append(float(np.linalg.norm(ef)))
        dists.append(float(np.linalg.norm(x_true[3:6] - x_true[0:3])))
        Pf = 0.5 * (P[3:6, 3:6] + P[3:6, 3:6].T)
        neess.append(float(ef @ np.linalg.solve(Pf + 1e-9 * np.eye(3), ef)))
    errs = np.array(errs)
    return (
        errs[-1],
        float(np.sqrt((errs**2).mean())),
        float(np.median(neess[20:])),
        bool(errs[-1] < 5),
        float(np.mean(dists)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--mag", type=float, default=2.26)
    ap.add_argument(
        "--metric",
        default="neg_softmin_eig",
        choices=list(metrics.METRICS),
        help="neg_logdet drives-away (volume-max); neg_softmin_eig is the stable choice",
    )
    args = ap.parse_args()
    global METRIC  # noqa: PLW0603 -- experiment knob: select the observability metric at CLI
    METRIC = metrics.METRICS[args.metric]

    dirs = {
        "radial": args.mag * RHAT,
        "tang_horiz": args.mag * THAT_H,
        "tang_vert": args.mag * THAT_V,
    }
    ctrls = {
        "noac": None,
        "weave": None,
        "oac": build_oac(),
        "oac_tight": build_oac(dist_bounds=(4.5, 5.5)),
        "hybrid": build_hybrid(),
    }

    print(
        f"3D point-mass OAC validation (rung 2): {args.seeds} seeds, |err0|={args.mag:.2f} m, "
        f"Lie order {ORDER} (NEES dof=3; stable=final<5 m)\n"
    )
    for dname, off in dirs.items():
        print(f"--- initial error: {dname} {np.round(off, 2)} ---")
        print(
            f"{'mode':>9} {'final_med':>10} {'final_p90':>10} {'rmse_med':>9} {'NEES_med':>9} "
            f"{'%stable':>8} {'meandist':>9}"
        )
        base = None
        for mode, ctrl in ctrls.items():
            R = np.array([run_once(mode, ctrl, s, off) for s in range(args.seeds)])
            fe, rm, ne, st, di = R[:, 0], R[:, 1], R[:, 2], R[:, 3], R[:, 4]
            if mode == "noac":
                base = np.median(fe)
            tag = "" if mode == "noac" else f"  ({base / np.median(fe):.1f}x)"
            print(
                f"{mode:>9} {np.median(fe):>10.3f} {np.percentile(fe, 90):>10.3f} {np.median(rm):>9.3f} "
                f"{np.median(ne):>9.1f} {100 * np.mean(st):>7.0f}% {np.median(di):>9.2f}{tag}"
            )
        print()


if __name__ == "__main__":
    main()
