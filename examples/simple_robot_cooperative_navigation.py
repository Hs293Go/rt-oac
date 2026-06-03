"""Headline example: real-time planar leader-follower cooperative navigation with EKF.

The future-work counterpart of the companion repo's
``examples/simple_robot_cooperative_navigation.py``. Same planar unicycle leader-follower
(``leader_follower_robots``; the follower's position is observable only through the
inter-robot range), with a **carried** EKF estimate (the controller sees only the estimate,
truth evolves separately, error accumulates) -- unlike the companion example, which
re-anchored the estimate to truth each step.

Frontier method: **log-det** objective + **SLSQP** early-stopped at 6 iters. Compares
observability-aware control against a no-OAC follower (drives straight) on closed-loop EKF
estimation accuracy, from a large initial follower-position error. Validated result:
OAC drives the unobservable initial error down (~13x lower final error, ~20x tighter
covariance) where no-OAC cannot, at ~3 ms/solve. See ``PROGRESS.md`` / ``results/RESULTS.md``.

Run:  uv run python examples/simple_robot_cooperative_navigation.py
"""

import pathlib
import time

from example_lib.misc import simple_ekf
from example_lib.models import leader_follower_robots as mdl
import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
from observability_aware_control import integrator, observability_cost

import rt_oac  # noqa: F401
from rt_oac import metrics
from rt_oac.controller import RTController

mpl.use("Agg")
import matplotlib.pyplot as plt

DT = 0.1
STLOG_DT = 0.2
WINDOW = 5
VAR = np.r_[np.full(2, 1e-1), np.full(2, 1e-2), 1e-2]  # [leader pos, headings, range]
INPUT_VAR = np.tile([1e-2, 1e-2], 2)
X0 = np.array([0.5, 0.0, 0.0, 0.0, 5.0, 0.0])
LEADER_U = np.array([1.0, 0.0])
STEPS = 120
SEED = 0
INIT_FOLLOWER_OFFSET = np.array([0.0, 0.0, 0.0, 1.6, 1.6, 0.0])  # ~2.3 m initial error


def build_controller():
    cost = observability_cost.ObservabilityCost(
        mdl.dynamics,
        mdl.observation,
        DT,
        gramian_kw={"order": 1, "var": VAR},
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
        maxiter=6,
        constraint=mdl.interrobot_distance,
        constraint_bounds=(0.2, 6.0),
    )


def run(use_oac, ctrl):
    rng = np.random.default_rng(SEED)
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
    P = np.diag([1e-3, 1e-3, 1e-3, 2.0, 2.0, 1e-2])
    x_hat = x_true + INIT_FOLLOWER_OFFSET
    err, sig, walls = [], [], []
    prev_u = None
    for _ in range(STEPS):
        if use_oac:
            guess = np.tile(np.r_[LEADER_U, 1.0, 0.0], (WINDOW, 1))
            if prev_u is not None:
                guess[:, 2:4] = np.vstack([prev_u[1:, 2:4], prev_u[-1:, 2:4]])
            t = time.perf_counter()
            res = ctrl.solve(x_hat, guess)  # controller acts on the ESTIMATE
            walls.append((time.perf_counter() - t) * 1e3)
            prev_u = res.u
            u_foll = res.u[0, 2:4]
        else:
            u_foll = np.array([1.0, 0.0])  # drive straight (no observability seeking)
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
        err.append(float(np.linalg.norm(x_hat[3:5] - x_true[3:5])))
        sig.append(float(3 * np.sqrt(max(np.trace(P[3:5, 3:5]) / 2, 0))))
    err = np.array(err)
    return {
        "err": err,
        "sig": np.array(sig),
        "rmse": float(np.sqrt((err**2).mean())),
        "final": float(err[-1]),
        "walls": walls,
    }


def main():
    ctrl = build_controller()
    oac = run(True, ctrl)
    noac = run(False, ctrl)
    tt = np.arange(STEPS) * DT

    print("=== RT-OAC planar cooperative navigation (frontier OPC + carried EKF) ===")
    print("follower-position estimation error  |   no-OAC   |    OAC")
    print(
        f"  RMSE  [m]                          | {noac['rmse']:8.3f}   | {oac['rmse']:8.3f}"
    )
    print(
        f"  final [m]                          | {noac['final']:8.3f}   | {oac['final']:8.3f}"
    )
    print(
        f"  median solve time [ms]             |     --     | {np.median(oac['walls']):8.1f}"
    )
    print("=> OAC drives the unobservable initial error down; no-OAC cannot.")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.plot(tt, noac["err"], color="tab:gray", lw=2, label="no-OAC error")
    a1.plot(tt, oac["err"], color="tab:red", lw=2, label="OAC error")
    a1.plot(
        tt, oac["sig"], color="tab:red", lw=1, ls="--", alpha=0.6, label="OAC 3-sigma"
    )
    a1.set(title="Follower-position estimation error", xlabel="t [s]", ylabel="m")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=8)
    a2.bar(
        ["no-OAC", "OAC"], [noac["rmse"], oac["rmse"]], color=["tab:gray", "tab:red"]
    )
    a2.set(title="follower-position RMSE [m]", ylabel="m")
    a2.grid(alpha=0.3, axis="y")
    fig.suptitle(
        "RT-OAC planar: log-det + SLSQP@6 + carried EKF (OAC vs no-OAC)", fontsize=12
    )
    fig.tight_layout()
    out = pathlib.Path(__file__).resolve().parents[1] / "results" / "example_planar.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
