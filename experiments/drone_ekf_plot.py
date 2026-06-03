"""Drone closed-loop: trajectory + EKF covariance, and the cost of the distance constraint.

Frontier controller (log-det + SLSQP + early-stop@6) with the inter-drone distance
constraint, a carried EKF estimate (Euclidean SimpleEKF + per-step quaternion renorm --
approximate; the rigorous error-state EKF is future work). Plots the relative trajectory
(follower about the leader), the EKF relative-position error + 3-sigma, and the inter-drone
distance vs its [1,3] m bounds. Also times the solve WITH vs WITHOUT the constraint to show
why enforcing it does not blow up the solve.

Left in the working tree on purpose.
"""

import pathlib
import time

from example_lib.misc import simple_ekf
from example_lib.models import inter_quadrotor_pose as mdl
import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
from observability_aware_control import integrator

import rt_oac  # noqa: F401
from rt_oac import metrics
from rt_oac.controller import RTController
from rt_oac.scenario import build_rt_controller, build_scenario
from rt_oac.warmstart import warm_guess

mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def quat_z(deg):
    a = np.deg2rad(deg) / 2
    return np.array([0.0, 0.0, np.sin(a), np.cos(a)])


def renorm(x):
    x = np.array(x)
    x[3:7] /= np.linalg.norm(x[3:7])
    return x


def main():
    sc = build_scenario(gramian_metric=metrics.neg_logdet)
    ctrl = build_rt_controller(
        sc, method="SLSQP", maxiter=6
    )  # WITH distance constraint
    dt = sc.cfg["sim"]["integrator_dt"]
    var = np.concatenate([
        [sc.cfg["noise"]["range_var"]],
        np.full(4, sc.cfg["noise"]["att_var"]),
    ])
    input_var = np.tile([0.05, 0.01, 0.01, 0.01], 2)
    lo = sc.cfg["opc"]["min_inter_vehicle_distance"]
    hi = sc.cfg["opc"]["max_inter_vehicle_distance"]

    sim = jax.jit(
        integrator.Integrator(mdl.dynamics, integrator.Methods.EULER, stepsize=dt)
    )
    ekf = simple_ekf.SimpleEKF(
        lambda x, u, dt: x + dt * mdl.dynamics(x, u),
        lambda x: mdl.observation(x),
        in_cov=np.diag(input_var),
        obs_cov=np.diag(var),
    )

    rng = np.random.default_rng(0)
    x_true = np.concatenate([[2.0, 0.0, 0.0], quat_z(20), [0.0, 1.0, 0.0]])  # non-MUC
    P = np.diag([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3, 1e-3, 0.4, 0.4, 0.4])
    x_hat = renorm(x_true + np.r_[0.9, 0.9, 0.4, 0, 0, 0, 0, 0.1, 0.1, 0.0])

    steps = 60
    rec = {"rel": [], "err": [], "sig": [], "dist": []}
    prev_u = None
    for i in range(steps):
        guess = warm_guess(sc.reference_guess(i), prev_u)
        res = ctrl.solve(x_hat, guess)
        prev_u = res.u
        u = res.u[0].copy()
        u[4:] += rng.normal(0, np.sqrt(input_var[4:]))  # follower input noise

        x_hat, P = ekf.predict(jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(u), dt)
        x_hat = renorm(x_hat)
        x_true = renorm(sim(jnp.asarray(x_true), jnp.asarray(u))[0])
        y = np.array(mdl.observation(jnp.asarray(x_true))) + rng.normal(0, np.sqrt(var))
        x_hat, P = ekf.update(jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(y))
        x_hat, P = renorm(x_hat), np.array(P)

        rec["rel"].append(x_true[0:3].copy())
        rec["err"].append(float(np.linalg.norm(x_hat[0:3] - x_true[0:3])))
        rec["sig"].append(float(3 * np.sqrt(max(np.trace(P[0:3, 0:3]) / 3, 0))))
        rec["dist"].append(float(np.linalg.norm(x_true[0:3])))

    rel = np.array(rec["rel"])
    tt = np.arange(steps) * dt

    # ---- constraint cost: WITH vs WITHOUT the distance constraint -----------------
    ctrl_noc = RTController(
        sc.cost,
        stlog_dt=sc.stlog_dt,
        lb=np.array(sc.cfg["optim"]["lb"]),
        ub=np.array(sc.cfg["optim"]["ub"]),
        n_inputs=8,
        follower_indices=(4, 5, 6, 7),
        method="SLSQP",
        maxiter=6,
        constraint=None,
    )
    x_probe = np.concatenate([[2.0, 0.0, 0.0], quat_z(20), [0.0, 1.0, 0.0]])
    g0 = sc.reference_guess(0)

    def timeit(c):
        c.solve(x_probe, g0)  # warmup
        ts = []
        for _ in range(8):
            t = time.perf_counter()
            r = c.solve(x_probe, g0)
            ts.append((time.perf_counter() - t) * 1e3)
        return float(np.median(ts)), r.nit

    with_ms, with_nit = timeit(ctrl)
    noc_ms, noc_nit = timeit(ctrl_noc)
    print(f"drone solve WITH constraint:    {with_ms:6.1f} ms  (nit {with_nit})")
    print(f"drone solve WITHOUT constraint: {noc_ms:6.1f} ms  (nit {noc_nit})")
    print(
        f"constraint overhead: {with_ms - noc_ms:+.1f} ms ({100 * (with_ms - noc_ms) / with_ms:+.0f}%)"
    )
    print(
        f"final EKF rel-pos error {rec['err'][-1]:.3f} m, 3-sigma {rec['sig'][-1]:.3f} m, "
        f"dist range [{min(rec['dist']):.2f},{max(rec['dist']):.2f}] m"
    )

    # ---- plot --------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    sc3 = ax.scatter(rel[:, 0], rel[:, 1], rel[:, 2], c=tt, cmap="viridis", s=14)
    ax.plot(rel[:, 0], rel[:, 1], rel[:, 2], color="0.6", lw=0.8)
    ax.scatter([0], [0], [0], marker="*", c="black", s=200, label="leader")
    ax.scatter(
        *rel[0], marker="o", edgecolor="k", facecolor="none", s=90, label="start"
    )
    fig.colorbar(sc3, ax=ax, pad=0.1, label="time [s]")
    ax.set(
        title="Follower trajectory in leader-relative frame",
        xlabel="x [m]",
        ylabel="y [m]",
        zlabel="z [m]",
    )
    ax.legend(loc="upper left", fontsize=8)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(tt, rec["err"], "-", color="tab:red", lw=2, label="EKF rel-pos error [m]")
    ax2.plot(tt, rec["sig"], "-", color="0.6", lw=1.5, label="3-sigma bound [m]")
    ax2.set(title="EKF relative-position uncertainty", ylabel="m")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(2, 2, 4)
    ax3.plot(tt, rec["dist"], "-", color="tab:blue", lw=2, label="inter-drone distance")
    ax3.axhline(lo, color="r", ls="--", lw=1, label="min/max bound")
    ax3.axhline(hi, color="r", ls="--", lw=1)
    ax3.set(title="Inter-drone distance vs constraint", xlabel="time [s]", ylabel="m")
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=8)

    fig.suptitle(
        "Drone case: log-det + SLSQP@6 + distance constraint, with carried EKF",
        fontsize=13,
    )
    fig.tight_layout()
    out = pathlib.Path(__file__).resolve().parents[1] / "results" / "drone_ekf.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
