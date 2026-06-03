"""Drone receding-horizon PLANNING with perfect state feedback (no EKF).

The OPC sees the true relative state each tick, solves (log-det + SLSQP + early-stop@6 +
distance constraint, warm-started), applies the first control, and advances the true
dynamics. This isolates the planner from the estimator. Reports per-tick planning time and
whether the observability-optimal orbiting behavior is preserved (relative-frame trajectory
+ cumulative bearing sweep). Left in the working tree on purpose.
"""

import pathlib
import time

from example_lib.models import inter_quadrotor_pose as mdl
import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
from observability_aware_control import integrator

import rt_oac  # noqa: F401
from rt_oac import metrics
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
    ctrl = build_rt_controller(sc, method="SLSQP", maxiter=6)
    dt = sc.cfg["sim"]["integrator_dt"]
    lo = sc.cfg["opc"]["min_inter_vehicle_distance"]
    hi = sc.cfg["opc"]["max_inter_vehicle_distance"]
    sim = jax.jit(
        integrator.Integrator(mdl.dynamics, integrator.Methods.EULER, stepsize=dt)
    )

    x = np.concatenate([[2.0, 0.0, 0.0], quat_z(20), [0.0, 1.0, 0.0]])  # non-MUC
    steps = 100
    rel, walls, nits, dist = [], [], [], []
    prev_u = None
    for i in range(steps):
        guess = warm_guess(sc.reference_guess(i), prev_u)
        t = time.perf_counter()
        res = ctrl.solve(x, guess)  # PERFECT feedback: planner sees true state x
        walls.append((time.perf_counter() - t) * 1e3)
        prev_u = res.u
        nits.append(res.nit)
        rel.append(x[0:3].copy())
        dist.append(float(np.linalg.norm(x[0:3])))
        x = renorm(sim(jnp.asarray(x), jnp.asarray(res.u[0]))[0])

    rel = np.array(rel)
    tt = np.arange(steps) * dt
    steady = slice(1, None)  # drop tick-0 compile
    az = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
    sweep = np.degrees(az[-1] - az[0])
    print(
        f"planning time: median {np.median(walls[steady]):.0f} ms  p95 {np.percentile(walls[steady], 95):.0f} ms  "
        f"(tick0 w/ compile {walls[0]:.0f} ms)"
    )
    print(f"iterations: median {int(np.median(nits))}")
    print(
        f"orbiting: cumulative bearing sweep {sweep:+.0f} deg ({sweep / 360:+.2f} revolutions)"
    )
    print(
        f"inter-drone distance range [{min(dist):.2f}, {max(dist):.2f}] m  (bound [{lo},{hi}])"
    )

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    s = ax.scatter(rel[:, 0], rel[:, 1], rel[:, 2], c=tt, cmap="viridis", s=14)
    ax.plot(rel[:, 0], rel[:, 1], rel[:, 2], color="0.6", lw=0.8)
    ax.scatter([0], [0], [0], marker="*", c="black", s=200, label="leader")
    ax.scatter(
        *rel[0], marker="o", edgecolor="k", facecolor="none", s=90, label="start"
    )
    fig.colorbar(s, ax=ax, pad=0.1, label="time [s]")
    ax.set(
        title="Follower trajectory in leader-relative frame\n(perfect-feedback planning)",
        xlabel="x [m]",
        ylabel="y [m]",
        zlabel="z [m]",
    )
    ax.legend(loc="upper left", fontsize=8)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(tt, dist, "-", color="tab:blue", lw=2, label="inter-drone distance")
    ax2.axhline(lo, color="r", ls="--", lw=1, label="bound")
    ax2.axhline(hi, color="r", ls="--", lw=1)
    ax2.set(title="Inter-drone distance vs constraint", ylabel="m")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(2, 2, 4)
    ax3.plot(tt, walls, "-", color="tab:green", lw=1.5)
    ax3.axhline(100, color="0.6", ls=":", lw=1, label="100 ms (10 Hz)")
    ax3.set(title="Per-tick planning time", xlabel="time [s]", ylabel="ms")
    ax3.set_ylim(0, max(np.percentile(walls, 98), 130))
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=8)

    fig.suptitle(
        "Drone receding-horizon planning, perfect feedback (no EKF): log-det + SLSQP@6",
        fontsize=13,
    )
    fig.tight_layout()
    out = pathlib.Path(__file__).resolve().parents[1] / "results" / "drone_openloop.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
