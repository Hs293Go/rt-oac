r"""Rigorously validate the planar beneficial-OAC result (the bottom of the bottom-up ladder).

The headline planar example shows OAC ~5x lower final follower-position error than a no-OAC
(drive-straight) follower, but on a SINGLE seed + a single fixed initial-error direction. This sweeps
seed x initial-error-direction x control-mode to test (a) is the OAC benefit ROBUST, (b) is the loop
STABLE across seeds, (c) WHERE does OAC help -- the prediction is: most on the TANGENTIAL error (range-
only unobservable without motion), least on the RADIAL (observed by range even driving straight).

    PYTHONPATH=src JAX_PLATFORMS=cpu uv run python experiments/planar_oac_validate.py --seeds 20
"""

import argparse

from example_lib.misc import simple_ekf
from example_lib.models import leader_follower_robots as mdl
import jax
import jax.numpy as jnp
import numpy as np
from observability_aware_control import integrator, observability_cost

from rt_oac import metrics, tracking_cost
from rt_oac.balanced_cost import make_balanced
from rt_oac.controller import RTController

DT, STLOG_DT, WINDOW, STEPS = 0.1, 0.2, 5, 120
VAR = np.r_[np.full(2, 1e-1), np.full(2, 1e-2), 1e-2]  # [leader pos, headings, range]
INPUT_VAR = np.tile([1e-2, 1e-2], 2)
X0 = np.array([0.5, 0.0, 0.0, 0.0, 5.0, 0.0])  # [leader xyth, follower xyth]
LEADER_U = np.array([1.0, 0.0])
DIST_BOUNDS = (0.2, 6.0)
P0 = np.diag([1e-3, 1e-3, 1e-3, 2.0, 2.0, 1e-2])
FORMATION_OFFSET = X0[3:5] - X0[0:2]
HYBRID_RHO, HYBRID_WEIGHT = 1.0, 1.0

# range direction (leader->follower) and its tangent, for directional initial-error offsets
_r = X0[3:5] - X0[0:2]
RHAT = _r / np.linalg.norm(_r)  # radial (observed by range)
THAT = np.array([-RHAT[1], RHAT[0]])  # tangential (motion-observable only)


def _cost():
    return observability_cost.ObservabilityCost(
        mdl.dynamics,
        mdl.observation,
        DT,
        gramian_kw={"order": 1, "var": VAR},
        gramian_metric=metrics.neg_logdet,
        observed_indices=(),
    )


def build_oac(dist_bounds=DIST_BOUNDS):
    cost = _cost()
    ctrl = RTController(
        cost,
        stlog_dt=STLOG_DT,
        lb=np.array([0.0, -2.0]),
        ub=np.array([4.0, 2.0]),
        n_inputs=4,
        follower_indices=(2, 3),
        method="SLSQP",
        maxiter=6,
        constraint=mdl.interrobot_distance,
        constraint_bounds=dist_bounds,
        constraint_mode="hard",
    )
    return ctrl


def build_hybrid():
    cost = _cost()
    ref_us = np.tile(np.r_[LEADER_U, 1.0, 0.0], (WINDOW, 1))
    s_obs = abs(float(cost(jnp.asarray(X0), jnp.asarray(ref_us), STLOG_DT).objective))
    track = tracking_cost.quadratic_tracking_cost(position_indices=(3, 4), w_pos=1.0)
    bal = make_balanced(
        cost, track, scheme="normalized", s_track=HYBRID_RHO**2 * WINDOW, s_obs=s_obs
    )
    ctrl = RTController(
        bal,
        stlog_dt=STLOG_DT,
        lb=np.array([0.0, -2.0]),
        ub=np.array([4.0, 2.0]),
        n_inputs=4,
        follower_indices=(2, 3),
        method="SLSQP",
        maxiter=6,
        constraint=mdl.interrobot_distance,
        constraint_bounds=DIST_BOUNDS,
        constraint_mode="hard",
    )
    return ctrl


def leader_pred(leader_state):
    px, py, th = leader_state
    ks = np.arange(WINDOW)
    return np.stack(
        [
            px + ks * DT * LEADER_U[0] * np.cos(th),
            py + ks * DT * LEADER_U[0] * np.sin(th),
        ],
        axis=1,
    )


def run_once(mode, ctrl, seed, offset):
    """One closed-loop run. mode in {noac, oac, hybrid}. Returns (final_err, rmse, nees_med, stable)."""
    rng = np.random.default_rng(seed)
    sim = jax.jit(
        integrator.Integrator(mdl.dynamics, integrator.Methods.RK4, stepsize=DT)
    )
    ekf = simple_ekf.SimpleEKF(
        lambda x, u, dt: x + dt * mdl.dynamics(x, u),
        lambda x: mdl.observation(x, 0),
        in_cov=np.diag(INPUT_VAR),
        obs_cov=np.diag(VAR),
    )
    x_true = X0.copy()
    P = P0.copy()
    x_hat = x_true + np.r_[0, 0, 0, offset, 0.0]
    prev_u, errs, neess, dists = None, [], [], []
    for i in range(STEPS):
        if mode == "noac":
            u_foll = np.array([1.0, 0.0])  # drive straight (no excitation)
        elif mode == "weave":
            u_foll = np.array([
                1.0,
                1.0 * np.sin(2 * np.pi * i / 15),
            ])  # dumb excitation (not OAC)
        else:
            guess = np.tile(np.r_[LEADER_U, 1.0, 0.0], (WINDOW, 1))
            if prev_u is not None:
                guess[:, 2:4] = np.vstack([prev_u[1:, 2:4], prev_u[-1:, 2:4]])
            if mode == "hybrid":
                p_ref = leader_pred(x_hat[0:3]) + FORMATION_OFFSET
                res = ctrl.solve(x_hat, guess, p_ref=p_ref, weight=HYBRID_WEIGHT)
            else:
                res = ctrl.solve(x_hat, guess)
            prev_u = res.u
            u_foll = np.asarray(res.u[0, 2:4])
        u = np.r_[LEADER_U, u_foll]
        u_applied = u + np.r_[0, 0, rng.normal(0, np.sqrt(INPUT_VAR[2:]))]
        x_hat, P = ekf.predict(
            jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(u_applied), DT
        )
        x_true = np.array(sim(jnp.asarray(x_true), jnp.asarray(u_applied))[0])
        y = np.array(mdl.observation(jnp.asarray(x_true), 0)) + rng.normal(
            0, np.sqrt(VAR)
        )
        x_hat, P = ekf.update(jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(y))
        x_hat, P = np.array(x_hat), np.array(P)
        ef = x_hat[3:5] - x_true[3:5]
        errs.append(float(np.linalg.norm(ef)))
        dists.append(float(np.linalg.norm(x_true[0:2] - x_true[3:5])))
        Pf = 0.5 * (P[3:5, 3:5] + P[3:5, 3:5].T)
        neess.append(float(ef @ np.linalg.solve(Pf + 1e-9 * np.eye(2), ef)))
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
    ap.add_argument(
        "--mag",
        type=float,
        default=2.26,
        help="initial follower-position error magnitude (m)",
    )
    args = ap.parse_args()

    dirs = {
        "tangential": args.mag * THAT,
        "radial": args.mag * RHAT,
        "diagonal": np.array([1.6, 1.6]),
    }
    # oac_tight: distance pinned near the no-OAC standoff (~5 m) so OAC can ONLY maneuver laterally,
    # not approach -- isolates the observability-MANEUVER benefit from the get-closer (range-SNR) benefit.
    ctrls = {
        "noac": None,
        "weave": None,
        "oac": build_oac(),
        "oac_tight": build_oac(dist_bounds=(4.5, 5.5)),
        "hybrid": build_hybrid(),
    }

    print(
        f"planar OAC validation: {args.seeds} seeds, |err0|={args.mag:.2f} m "
        f"(NEES dof=2; stable = final<5 m)\n"
    )
    for dname, off in dirs.items():
        print(f"--- initial error: {dname} {np.round(off, 2)} ---")
        print(
            f"{'mode':>8} {'final_med':>10} {'final_p90':>10} {'rmse_med':>9} {'NEES_med':>9} "
            f"{'%stable':>8} {'meandist':>9}"
        )
        base = None
        for mode, ctrl in ctrls.items():
            R = np.array([run_once(mode, ctrl, s, off) for s in range(args.seeds)])
            fe, rm, ne, st, di = R[:, 0], R[:, 1], R[:, 2], R[:, 3], R[:, 4]
            if mode == "noac":
                base = np.median(fe)
            tag = "" if mode == "noac" else f"  ({base / np.median(fe):.1f}x vs noac)"
            print(
                f"{mode:>8} {np.median(fe):>10.3f} {np.percentile(fe, 90):>10.3f} {np.median(rm):>9.3f} "
                f"{np.median(ne):>9.1f} {100 * np.mean(st):>7.0f}% {np.median(di):>9.2f}{tag}"
            )
        print()


if __name__ == "__main__":
    main()
