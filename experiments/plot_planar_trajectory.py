"""Plot the planar leader-follower 2D trajectory under OAC.

Leader drives a straight line; the follower runs the observability-aware controller
(log-det, SLSQP). We roll out the *true-state* closed loop (controller sees the true state)
so the plot reflects the OAC's geometric intent, and record both robots' positions.

Two panels:
  * world frame: absolute leader & follower paths (OAC vs a no-OAC straight baseline);
  * leader-relative frame: follower position minus leader position -- circling the leader
    shows up here as an actual loop around the origin.

Also prints the cumulative relative-bearing change (a quantitative "circling" measure).
Left in the working tree on purpose.
"""

import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

import rt_oac  # noqa: F401

# reuse the exact planar setup from the EKF eval
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from example_lib.models import leader_follower_robots as mdl
import matplotlib as mpl
from observability_aware_control import integrator
from planar_ekf_eval import DT, LEADER_U, WINDOW, X0, build_controller

mpl.use("Agg")
import matplotlib.pyplot as plt


def rollout(condition, steps, maxiter=20):
    sim = jax.jit(
        integrator.Integrator(mdl.dynamics, integrator.Methods.RK4, stepsize=DT)
    )
    ctrl = build_controller(maxiter) if condition == "OAC" else None
    x = X0.copy()
    traj = [x.copy()]
    prev_u = None
    follower_guess = np.array([1.0, 0.0])
    for _ in range(steps):
        if condition == "OAC":
            guess = np.tile(np.r_[LEADER_U, follower_guess], (WINDOW, 1))
            if prev_u is not None:
                guess[:, 2:4] = np.vstack([prev_u[1:, 2:4], prev_u[-1:, 2:4]])
            res = ctrl.solve(x, guess)  # controller sees the TRUE state
            prev_u = res.u
            u_foll = res.u[0, 2:4]
        else:
            u_foll = follower_guess
        u = np.r_[LEADER_U, u_foll]
        x_next, _ = sim(jnp.asarray(x), jnp.asarray(u))
        x = np.array(x_next)
        traj.append(x.copy())
    return np.array(traj)  # (steps+1, 6): leader [0:3], follower [3:6]


def bearing_change(traj):
    rel = traj[:, 3:5] - traj[:, 0:2]
    ang = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
    return ang - ang[0]


def main():
    steps = 150
    oac = rollout("OAC", steps, maxiter=20)
    noac = rollout("no-OAC", steps)

    lead = oac[:, 0:2]
    f_oac = oac[:, 3:5]
    f_noac = noac[:, 3:5]
    rel_oac = f_oac - lead
    rel_noac = f_noac - lead

    db_oac = bearing_change(oac)
    db_noac = bearing_change(noac)
    print(
        f"cumulative relative-bearing change (rad):  OAC {db_oac[-1]:+.2f} "
        f"({np.degrees(db_oac[-1]):+.0f} deg)   no-OAC {db_noac[-1]:+.2f} "
        f"({np.degrees(db_noac[-1]):+.0f} deg)"
    )
    print(
        f"total |bearing| swept (rad):               OAC {np.abs(np.diff(db_oac)).sum():.2f}"
        f"   no-OAC {np.abs(np.diff(db_noac)).sum():.2f}"
    )
    print(
        f"inter-robot distance OAC: min {np.linalg.norm(rel_oac, axis=1).min():.2f} "
        f"max {np.linalg.norm(rel_oac, axis=1).max():.2f} m"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # world frame
    t = np.arange(len(oac))
    ax1.plot(
        lead[:, 0], lead[:, 1], "-", color="black", lw=2, label="leader (straight)"
    )
    ax1.plot(
        f_noac[:, 0],
        f_noac[:, 1],
        "--",
        color="gray",
        lw=1.5,
        label="follower (no-OAC)",
    )
    sc = ax1.scatter(
        f_oac[:, 0], f_oac[:, 1], c=t, cmap="viridis", s=10, label="follower (OAC)"
    )
    ax1.scatter(*lead[0], marker="o", c="black", s=80, zorder=5)
    ax1.scatter(
        *f_oac[0],
        marker="o",
        edgecolor="k",
        facecolor="none",
        s=90,
        zorder=5,
        label="start",
    )
    ax1.scatter(*f_oac[-1], marker="s", c="red", s=70, zorder=5, label="OAC end")
    fig.colorbar(sc, ax=ax1, label="time step")
    ax1.set(title="World frame", xlabel="x [m]", ylabel="y [m]")
    ax1.axis("equal")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="best", fontsize=8)

    # leader-relative frame (does the follower loop around the leader?)
    ax2.axhline(0, color="0.85", lw=0.8)
    ax2.axvline(0, color="0.85", lw=0.8)
    ax2.plot(rel_noac[:, 0], rel_noac[:, 1], "--", color="gray", lw=1.5, label="no-OAC")
    sc2 = ax2.scatter(
        rel_oac[:, 0], rel_oac[:, 1], c=t, cmap="viridis", s=10, label="OAC"
    )
    ax2.scatter(0, 0, marker="*", c="black", s=200, zorder=5, label="leader")
    ax2.scatter(
        *rel_oac[0], marker="o", edgecolor="k", facecolor="none", s=90, zorder=5
    )
    fig.colorbar(sc2, ax=ax2, label="time step")
    ax2.set(
        title="Follower in leader-relative frame",
        xlabel="x rel. to leader [m]",
        ylabel="y rel. to leader [m]",
    )
    ax2.axis("equal")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle(
        "Planar leader-follower trajectory under observability-aware control",
        fontsize=13,
    )
    fig.tight_layout()
    out = (
        pathlib.Path(__file__).resolve().parents[1]
        / "results"
        / "planar_trajectory.png"
    )
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
