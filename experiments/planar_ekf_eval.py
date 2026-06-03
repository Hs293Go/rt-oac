"""Closed-loop EKF validation on the planar leader-follower (the decisive arbiter).

Does observability-aware control actually keep the estimator sharp -- and does the fast,
early-stopped solve preserve that benefit? The leader drives straight (v=1, w=0); the
follower's position is observable only through the inter-robot range, so it must maneuver to
stay observable. We carry a real EKF estimate across steps (the controller sees only the
estimate; truth evolves separately; error accumulates), unlike the companion example which
re-anchors to truth each step.

Conditions (identical seeds/scenario):
  * no-OAC   : follower drives straight (no observability optimization)
  * OAC-full : log-det objective, SLSQP maxiter=60
  * OAC-fast : log-det objective, SLSQP maxiter=6 (the ~100 ms real-time setting)

Metrics: follower-position RMSE, terminal EKF covariance (follower pos), solve time.
"""

import argparse

from example_lib.misc import simple_ekf
from example_lib.models import leader_follower_robots as mdl
import jax
import jax.numpy as jnp
import numpy as np
from observability_aware_control import integrator, observability_cost

import rt_oac  # noqa: F401
from rt_oac import metrics
from rt_oac.controller import RTController

NS = mdl.NUM_STATES  # 3 per robot
DT = 0.1
STLOG_DT = 0.2
WINDOW = 5
ORDER = 1

# observation noise: [leader_pos(2), headings(2), inter-robot range(1)]
VAR = np.r_[np.full(2, 1e-1), np.full(2, 1e-2), 1e-2]
INPUT_VAR = np.tile([1e-2, 1e-2], 2)  # per-robot (v, w)

X0 = np.array([0.5, 0.0, 0.0, 0.0, 5.0, 0.0])  # leader (0.5,0), follower (0,5)
LEADER_U = np.array([1.0, 0.0])  # straight at speed 1


def build_controller(maxiter):
    cost = observability_cost.ObservabilityCost(
        mdl.dynamics,
        mdl.observation,
        DT,
        gramian_kw={"order": ORDER, "var": VAR},
        gramian_metric=metrics.neg_logdet,
        observed_indices=(),
    )
    return RTController(
        cost,
        stlog_dt=STLOG_DT,
        lb=np.array([0.0, -2.0]),
        ub=np.array([4.0, 2.0]),
        n_inputs=4,
        follower_indices=(2, 3),
        method="SLSQP",
        maxiter=maxiter,
        constraint=mdl.interrobot_distance,
        constraint_bounds=(0.2, 6.0),
    )


def run(condition, steps, seed, maxiter=60, init_foll_var=4.0, ctrl=None):
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
    if ctrl is None and condition != "no-OAC":
        ctrl = build_controller(maxiter)

    x_true = X0.copy()
    # initial estimate: follower position poorly known (large initial error to stress
    # whether the estimator can drive it down -- only possible if observable)
    P = np.diag([1e-3, 1e-3, 1e-3, init_foll_var, init_foll_var, 1e-2])
    x_hat = x_true + rng.multivariate_normal(np.zeros(6), P)

    foll_err, walls, nits = [], [], []
    prev_u = None
    follower_guess = np.array([1.0, 0.0])  # straight as the reference guess
    for _step in range(steps):
        if condition == "no-OAC":
            u_foll = follower_guess
        else:
            guess = np.tile(np.r_[LEADER_U, follower_guess], (WINDOW, 1))
            if prev_u is not None:
                guess[:, 2:4] = np.vstack([prev_u[1:, 2:4], prev_u[-1:, 2:4]])
            res = ctrl.solve(x_hat, guess)  # controller sees the ESTIMATE
            prev_u = res.u
            u_foll = res.u[0, 2:4]
            walls.append(res.wall_time)
            nits.append(res.nit)
        u = np.r_[LEADER_U, u_foll]
        u_applied = u + np.r_[0, 0, rng.normal(0, np.sqrt(INPUT_VAR[2:]))]

        # EKF predict (uses applied control), true step, measure, EKF update
        x_hat, P = ekf.predict(
            jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(u_applied), DT
        )
        x_true, _ = sim(jnp.asarray(x_true), jnp.asarray(u_applied))
        x_true = np.array(x_true)
        y = np.array(mdl.observation(jnp.asarray(x_true), 0)) + rng.normal(
            0, np.sqrt(VAR)
        )
        x_hat, P = ekf.update(jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(y))
        x_hat, P = np.array(x_hat), np.array(P)

        foll_err.append(np.linalg.norm(x_hat[3:5] - x_true[3:5]))

    foll_err = np.array(foll_err)
    return {
        "condition": condition,
        "rmse_foll_pos": float(np.sqrt((foll_err**2).mean())),
        "final_foll_err": float(foll_err[-1]),
        "terminal_P_foll": float(np.trace(P[3:5, 3:5])),
        "median_wall_ms": float(np.median(walls) * 1e3) if walls else 0.0,
        "median_nit": float(np.median(nits)) if nits else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()

    print(f"# Planar EKF closed-loop eval (steps={args.steps}, seeds={args.seeds})")
    print(
        "# follower position observable only via inter-robot range; large initial error\n"
    )

    # build controllers once (reuse compiled artifacts across seeds)
    ctrl_full = build_controller(60)
    ctrl_fast = build_controller(6)
    conditions = [
        ("no-OAC", None, None),
        ("OAC-full", 60, ctrl_full),
        ("OAC-fast", 6, ctrl_fast),
    ]

    agg: dict[str, dict[str, list]] = {
        c[0]: {"rmse": [], "final": [], "P": [], "wall": [], "nit": []}
        for c in conditions
    }
    for seed in range(args.seeds):
        for name, mi, ctrl in conditions:
            r = run(name, args.steps, seed, maxiter=mi or 60, ctrl=ctrl)
            agg[name]["rmse"].append(r["rmse_foll_pos"])
            agg[name]["final"].append(r["final_foll_err"])
            agg[name]["P"].append(r["terminal_P_foll"])
            agg[name]["wall"].append(r["median_wall_ms"])
            agg[name]["nit"].append(r["median_nit"])

    print(
        f"{'condition':<12}{'RMSE(mean±std)':>20}{'final(mean)':>13}{'term P(mean)':>14}"
        f"{'wall ms':>10}{'nit':>6}"
    )
    for name, _, _ in conditions:
        a = agg[name]
        rmse = np.array(a["rmse"])
        print(
            f"{name:<12}{rmse.mean():>11.4f}±{rmse.std():<7.4f}"
            f"{np.mean(a['final']):>13.4f}{np.mean(a['P']):>14.4e}"
            f"{np.median(a['wall']):>10.0f}{np.median(a['nit']):>6.0f}"
        )


if __name__ == "__main__":
    main()
