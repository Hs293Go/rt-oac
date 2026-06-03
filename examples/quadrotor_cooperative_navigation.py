"""Headline example: real-time quadrotor cooperative-navigation OPC.

The future-work counterpart of the companion repo's
``examples/quadrotor_cooperative_navigation.py``. Same problem (relative-pose
``inter_quadrotor_pose`` model, straight-cruising leader, inter-drone distance constraint,
perfect-feedback online receding horizon), but with the RT-OAC frontier method:

  * objective: smooth **log-det** of the STLOG (vs the companion's degenerate per-node
    min-eigenvalue, which is floored by the ``T^(2 r* + 1)`` short-time scaling);
  * solver: **SLSQP** with **early stopping** at 6 iterations + warm-starting;

which solves in ~100 ms/tick (vs ~10 s for the companion's trust-constr + min-eig at order
5) while driving the trajectory to full observability. See ``PROGRESS.md`` and
``results/phase0_findings.md``.

This example uses *perfect state feedback* (the planner sees the true relative state), exactly
like the companion quadrotor example. Closing the loop with the manifold-aware
``ErrorStateEKF`` is consistent for gentle maneuvers but the aggressive observability orbit
couples unstably with estimation (PROGRESS.md finding #8) -- that is open future work, so the
headline here is the *planner*.

Run:  uv run python examples/quadrotor_cooperative_navigation.py
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

STEPS = 300  # x integrator_dt (0.05 s) = 15 s


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
    sim = jax.jit(
        integrator.Integrator(mdl.dynamics, integrator.Methods.EULER, stepsize=dt)
    )
    gram = jax.jit(
        lambda x, u: sc.cost(
            x, jnp.asarray(u), sc.stlog_dt, return_gramians=True
        ).gramians
    )

    x = np.concatenate([[2.0, 0.0, 0.0], quat_z(20), [0.0, 1.0, 0.0]])  # non-MUC start
    rel, walls, nobs, dist = [], [], [], []
    prev_u = None
    for i in range(STEPS):
        guess = warm_guess(sc.reference_guess(i), prev_u)
        t = time.perf_counter()
        res = ctrl.solve(x, guess)  # perfect feedback: planner sees true x
        walls.append((time.perf_counter() - t) * 1e3)
        prev_u = res.u
        g = np.asarray(gram(jnp.asarray(x), res.u))
        eig = np.linalg.eigvalsh(0.5 * (g + np.swapaxes(g, -1, -2)).sum(0))
        nobs.append(int((eig > 1e-6 * eig[-1]).sum()))
        rel.append(x[0:3].copy())
        dist.append(float(np.linalg.norm(x[0:3])))
        x = renorm(sim(jnp.asarray(x), jnp.asarray(res.u[0]))[0])

    rel = np.array(rel)
    steady = slice(1, None)
    print("=== RT-OAC quadrotor cooperative navigation (frontier OPC) ===")
    print(
        f"steps {STEPS} ({STEPS * dt:.0f} s) | median plan {np.median(np.array(walls)[steady]):.0f} ms"
        f" | p95 {np.percentile(np.array(walls)[steady], 95):.0f} ms (tick0 compile {walls[0]:.0f} ms)"
    )
    print(
        f"observable directions: median {int(np.median(nobs))}/6 | inter-drone distance "
        f"[{min(dist):.2f}, {max(dist):.2f}] m"
    )

    tt = np.arange(STEPS) * dt
    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    s = ax.scatter(rel[:, 0], rel[:, 1], rel[:, 2], c=tt, cmap="viridis", s=10)
    ax.plot(rel[:, 0], rel[:, 1], rel[:, 2], color="0.6", lw=0.7)
    ax.scatter([0], [0], [0], marker="*", c="black", s=180, label="leader")
    fig.colorbar(s, ax=ax, pad=0.1, label="t [s]")
    ax.set(
        title="Follower in leader-relative frame",
        xlabel="x [m]",
        ylabel="y [m]",
        zlabel="z [m]",
    )
    ax.legend(loc="upper left", fontsize=8)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(tt, walls, color="tab:green", lw=1.2)
    ax2.axhline(100, color="0.6", ls=":", lw=1, label="100 ms (10 Hz)")
    ax2.set(title="Per-tick plan time", xlabel="t [s]", ylabel="ms", ylim=(0, 250))
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.suptitle(
        "RT-OAC quadrotor: log-det + SLSQP@6, perfect-feedback receding horizon",
        fontsize=12,
    )
    fig.tight_layout()
    out = (
        pathlib.Path(__file__).resolve().parents[1]
        / "results"
        / "example_quadrotor.png"
    )
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
