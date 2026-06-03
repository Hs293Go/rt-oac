"""Headline example: real-time quadrotor cooperative-navigation OPC, with rerun.

The future-work counterpart of the companion repo's
``examples/quadrotor_cooperative_navigation.py``. Same problem (relative-pose
``inter_quadrotor_pose`` model, straight level constant-velocity leader, inter-drone distance
band [1, 2] m, perfect-feedback online receding horizon), but with the RT-OAC frontier method:

  * objective: smooth **soft-min eigenvalue** (E-optimality) of the STLOG -- a smooth
    surrogate for the companion's degenerate per-node min-eigenvalue (which is floored by the
    ``T^(2 r* + 1)`` short-time scaling). Maximizing the *worst* observable direction makes
    the follower settle into a clean planar observability orbit; the log-det (D-optimality)
    surrogate instead maximizes observability *volume* and explores a 3D shell. A narrow
    distance band bracketing the start radius is what turns the motion into a constant-radius
    ring rather than a baseline-maximizing arc -- see benchmarks/quad_constraint_sweep.py;
  * solver: **SLSQP** with **early stopping** at 6 iterations + warm-starting;

which solves in ~115 ms/tick on this CPU (measured median; vs ~10 s for the companion's
trust-constr + min-eig at order 5 -- a ~90x speedup, just over the 100 ms / 10 Hz target,
which the planar case clears outright) while driving the trajectory to full observability.
Soft-min is ~15% faster than log-det here because it orbits the band *interior*, leaving the
distance constraint inactive, whereas log-det pins the follower to the max-distance bound
(active constraint => costlier SLSQP QP). See ``PROGRESS.md``, ``results/phase0_findings.md``,
and benchmarks/{objective_solver_profile,objective_eval_microbench}.py.

This example uses *perfect state feedback* (the planner sees the true relative state), exactly
like the companion quadrotor example. Closing the loop with the manifold-aware
``ErrorStateEKF`` is consistent for gentle maneuvers but the aggressive observability orbit
couples unstably with estimation (PROGRESS.md finding #8) -- that is open future work, so the
headline here is the *planner*.

Visualization is **rerun**, mirroring the companion repo's examples: a live 3D scene of both
quadrotors flying (leader on its level cruise, follower reconstructed via
``to_absolute_state``) plus live time-series of plan time, observability, and the inter-drone
distance. A comprehensive multi-panel matplotlib figure is rendered after the run.

Live viewer:    uv run python examples/quadrotor_cooperative_navigation.py --spawn
Headless (default; writes results/example_quadrotor.rrd + .png):
                uv run python examples/quadrotor_cooperative_navigation.py
"""

import argparse
import dataclasses
import pathlib

from example_lib.models import inter_quadrotor_pose as mdl
from example_lib.visualization import visualization as viz
import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
from observability_aware_control import integrator
import rerun as rr
import rerun.blueprint as rrb
import tqdm

import rt_oac  # noqa: F401
from rt_oac import metrics
from rt_oac.controller import RTController
from rt_oac.scenario import build_scenario
from rt_oac.warmstart import warm_guess

mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

STEPS = 300  # x integrator_dt (0.05 s) = 15 s
# Distance band tightened to [1, 2] m (vs the paper's [1, 3]): a narrow band that brackets
# the start radius turns the motion into a clean constant-radius orbit instead of pegging at
# a far baseline. See benchmarks/quad_constraint_sweep.py for the band/objective sweep.
MIN_DIST, MAX_DIST = 1.0, 2.0
# Leader cruise speed. The paper's recorded leader is a min-snap 10 m / 120 s hop (peak
# ~0.16 m/s), so over this 15 s window it covers ~0.16 m -- visually frozen. The relative
# dynamics are Galilean-invariant (a common translation cancels in the relative state), so a
# faster *level* constant-velocity leader leaves the follower's orbit, observability, and
# solve time identical (verified in benchmarks/leader_speed_sweep.py) and only stretches the
# world-frame scene into a helix. 2 m/s covers ~30 m in 15 s -- a clear, realistic cruise.
LEADER_SPEED = 2.0
LEADER_ALT = 10.0
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"


def quat_z(deg):
    a = np.deg2rad(deg) / 2
    return np.array([0.0, 0.0, np.sin(a), np.cos(a)])


def level_cruise_leader(speed, n, dt, alt=LEADER_ALT):
    """A straight, level constant-velocity leader along +x (identity attitude, hover thrust).

    Returns (x_leader (n,10) [pos, quat xyzw, vel], u_leader (n,4) [thrust, body rates]).
    """
    t = np.arange(n) * dt
    x_leader = np.zeros((n, 10))
    x_leader[:, 0] = speed * t
    x_leader[:, 2] = alt
    x_leader[:, 6] = 1.0  # quaternion w (xyzw identity = level)
    x_leader[:, 7] = speed  # constant forward velocity
    u_leader = np.tile([9.81, 0.0, 0.0, 0.0], (n, 1))  # hover thrust, zero body rates
    return x_leader, u_leader


def renorm(x):
    x = np.array(x)
    x[3:7] /= np.linalg.norm(x[3:7])
    return x


def style_series(lo, hi):
    """Static styling for the rerun time-series (names/colors/widths)."""
    rr.log(
        "/graphs/solve/ms",
        rr.SeriesLines(names="plan time [ms]", colors=[40, 180, 80], widths=2),
        static=True,
    )
    rr.log(
        "/graphs/solve/budget",
        rr.SeriesLines(names="100 ms (10 Hz)", colors=[150, 150, 150], widths=1),
        static=True,
    )
    rr.log(
        "/graphs/observability/ndir",
        rr.SeriesLines(names="observable directions", colors=[40, 120, 230], widths=2),
        static=True,
    )
    rr.log(
        "/graphs/observability/min_eig",
        rr.SeriesLines(names="accumulated min-eig", colors=[180, 80, 220], widths=2),
        static=True,
    )
    rr.log(
        "/graphs/distance/dist",
        rr.SeriesLines(
            names="inter-drone distance [m]", colors=[230, 120, 40], widths=2
        ),
        static=True,
    )
    for name, _val in (("min", lo), ("max", hi)):
        rr.log(
            f"/graphs/distance/{name}",
            rr.SeriesLines(names=f"{name} bound", colors=[150, 150, 150], widths=1),
            static=True,
        )


def send_blueprint():
    rr.send_blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/sim", name="Cooperative quadrotors (world frame)"
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(origin="/graphs/solve", name="Plan time [ms]"),
                rrb.TimeSeriesView(
                    origin="/graphs/observability", name="Observability"
                ),
                rrb.TimeSeriesView(
                    origin="/graphs/distance", name="Inter-drone distance [m]"
                ),
                row_shares=[1, 1, 1],
            ),
            column_shares=[1.5, 1.0],
        )
    )


@jax.jit
def analyze_gramian_spectrum(g: jax.Array):
    acc = 0.5 * (g + g.swapaxes(-1, -2)).sum(0)
    eig = jnp.linalg.eigvalsh(acc)
    ndir = (eig > 1e-6 * eig[-1]).sum()
    mineig = eig[0]
    return ndir, mineig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--spawn", action="store_true", help="launch the live rerun viewer")
    ap.add_argument("--save", type=str, default=None, help="record to this .rrd path")
    args = ap.parse_args()

    sc = build_scenario(gramian_metric=metrics.neg_softmin_eig)
    # Replace the near-frozen min-snap leader with a level constant-velocity cruise so the
    # world-frame scene covers ground; the relative orbit is unchanged (Galilean invariance).
    n_lead = args.steps + sc.window + 1
    x_leader, u_leader = level_cruise_leader(
        LEADER_SPEED, n_lead, sc.cfg["sim"]["integrator_dt"]
    )
    sc = dataclasses.replace(sc, x_leader=x_leader, u_leader=u_leader)
    lo, hi = MIN_DIST, MAX_DIST
    ctrl = RTController(
        sc.cost,
        stlog_dt=sc.stlog_dt,
        lb=np.array(sc.cfg["optim"]["lb"]),
        ub=np.array(sc.cfg["optim"]["ub"]),
        n_inputs=8,
        follower_indices=(4, 5, 6, 7),
        method="SLSQP",
        maxiter=6,
        constraint=mdl.interrobot_distance_squared,
        constraint_bounds=(lo**2, hi**2),
    )
    dt = sc.cfg["sim"]["integrator_dt"]
    sim = jax.jit(
        integrator.Integrator(mdl.dynamics, integrator.Methods.EULER, stepsize=dt)
    )
    gram = jax.jit(
        lambda x, u: (
            sc.cost(x, jnp.asarray(u), sc.stlog_dt, return_gramians=True).gramians
        )
    )

    # --- rerun setup ---
    rr.init("rt_oac_quadrotor", spawn=args.spawn)
    rrd = args.save or str(RESULTS / "example_quadrotor.rrd")
    if not args.spawn:
        rr.save(rrd)
    style_series(lo, hi)
    send_blueprint()

    # Note: set time before initializing lines/frames
    rr.set_time("/time", duration=0)
    # Static world: the recorded leader path (so the live scene has context from t=0).
    lead_path = np.asarray(sc.x_leader[: args.steps, 0:3])
    rr.log(
        "/sim/leader/path",
        rr.LineStrips3D(lead_path[None], colors=[90, 90, 90]),
        static=True,
    )
    # Scene helpers: each logs its body-axis Arrows3D once and a per-step Transform3D at the
    # SAME entity path, so the axes (and the body frame) move with the pose; traces are
    # logged in world coordinates as siblings (not double-transformed).
    leader_viz = viz.PoseReferenceFrame("/sim/leader")
    leader_tr = viz.PositionTrace("/sim/leader", max_length=args.steps)
    foll_viz = viz.PoseReferenceFrame("/sim/follower")
    foll_tr = viz.PositionTrace("/sim/follower", max_length=args.steps)

    x = np.concatenate([[2.0, 0.0, 0.0], quat_z(20), [0.0, 1.0, 0.0]])  # non-MUC start
    rec = {
        k: [] for k in ("rel", "rq", "lead", "foll", "dist", "ndir", "mineig", "walls")
    }
    prev_u = None
    t = 0.0
    for i in tqdm.trange(args.steps):
        guess = warm_guess(sc.reference_guess(i), prev_u)
        res = ctrl.solve(x, guess)  # perfect feedback: planner sees true x
        prev_u = res.u

        # accumulated Gramian spectrum (over the prediction horizon)
        g = gram(jnp.asarray(x), res.u)
        ndir, mineig = analyze_gramian_spectrum(g)
        dist = float(jnp.linalg.norm(x[0:3]))

        # absolute world poses (leader on its recorded path; follower reconstructed)
        x_abs = np.asarray(mdl.to_absolute_state(sc.x_leader[i], jnp.asarray(x)))
        lead_p, lead_q = (
            np.asarray(sc.x_leader[i, 0:3]),
            np.asarray(sc.x_leader[i, 3:7]),
        )
        foll_p, foll_q = x_abs[10:13], x_abs[13:17]

        rec["rel"].append(x[0:3].copy())
        rec["rq"].append(np.array(x[3:7]))  # relative attitude (observed + optimized)
        rec["lead"].append(lead_p.copy())
        rec["foll"].append(foll_p.copy())
        rec["dist"].append(dist)
        rec["ndir"].append(int(ndir))
        rec["mineig"].append(float(mineig))
        rec["walls"].append(res.wall_time * 1e3)

        # --- rerun: 3D scene + metrics ---
        rr.set_time("/time", duration=t)
        leader_viz.set_pose(lead_p, rr.Quaternion(xyzw=lead_q))
        leader_tr.add_position(lead_p)
        foll_viz.set_pose(foll_p, rr.Quaternion(xyzw=foll_q))
        foll_tr.add_position(foll_p)
        rr.log(
            "/sim/range",
            rr.LineStrips3D(np.array([lead_p, foll_p])[None], colors=[120, 120, 120]),
        )
        rr.log("/graphs/solve/ms", rr.Scalars(res.wall_time * 1e3))
        rr.log("/graphs/solve/budget", rr.Scalars(100.0))
        rr.log("/graphs/observability/ndir", rr.Scalars(float(ndir)))
        rr.log("/graphs/observability/min_eig", rr.Scalars(mineig))
        rr.log("/graphs/distance/dist", rr.Scalars(dist))
        rr.log("/graphs/distance/min", rr.Scalars(lo))
        rr.log("/graphs/distance/max", rr.Scalars(hi))

        x = renorm(sim(jnp.asarray(x), jnp.asarray(res.u[0]))[0])
        t += dt

    summarize_and_plot(rec, dt, lo, hi, args.steps)
    if not args.spawn:
        print(f"rerun recording written; open with:  rerun {rrd}")


def _equal_3d(ax, pts):
    """Undistorted 3D view that fills the panel: box aspect proportional to the data
    extents (equal scale per axis, so a round orbit stays round) with limits set to the
    data range -- an elongated helix then fills an elongated box rather than a thin tube
    floating in a forced cube."""
    mins, maxs = pts.min(0), pts.max(0)
    ranges = np.maximum(maxs - mins, 1e-9)
    ax.set_box_aspect(ranges)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])


def summarize_and_plot(rec, dt, lo, hi, steps):
    rel = np.array(rec["rel"])
    rq = np.array(rec["rq"])  # relative attitude quaternion (xyzw)
    lead = np.array(rec["lead"])
    foll = np.array(rec["foll"])
    dist = np.array(rec["dist"])
    ndir = np.array(rec["ndir"])
    mineig = np.array(rec["mineig"])
    walls = np.array(rec["walls"])
    tt = np.arange(steps) * dt
    steady = slice(1, None)  # drop tick-0 compile

    print("=== RT-OAC quadrotor cooperative navigation (frontier OPC) ===")
    print(
        f"steps {steps} ({steps * dt:.0f} s) | median plan {np.median(walls[steady]):.0f} ms"
        f" | p95 {np.percentile(walls[steady], 95):.0f} ms (tick0 compile {walls[0]:.0f} ms)"
    )
    print(
        f"observable directions: median {int(np.median(ndir))}/6 | inter-drone distance "
        f"[{dist.min():.2f}, {dist.max():.2f}] m (bounds [{lo:.0f}, {hi:.0f}])"
    )

    fig = plt.figure(figsize=(20, 9))
    fig.suptitle(
        "RT-OAC quadrotor cooperative navigation: soft-min eig + SLSQP@6, perfect-feedback "
        "receding horizon",
        fontsize=14,
    )

    # (1) absolute world trajectory
    ax = fig.add_subplot(2, 4, 1, projection="3d")
    ax.plot(lead[:, 0], lead[:, 1], lead[:, 2], color="black", lw=2, label="leader")
    s = ax.scatter(foll[:, 0], foll[:, 1], foll[:, 2], c=tt, cmap="viridis", s=8)
    ax.scatter(*foll[0], c="tab:green", s=60, marker="o", label="follower start")
    ax.scatter(*foll[-1], c="tab:red", s=60, marker="X", label="follower end")
    fig.colorbar(s, ax=ax, pad=0.12, label="t [s]", shrink=0.6)
    ax.set(
        title="World-frame trajectory", xlabel="x [m]", ylabel="y [m]", zlabel="z [m]"
    )
    _equal_3d(ax, np.vstack([lead, foll]))
    ax.legend(loc="upper left", fontsize=7)

    # (2) relative-frame orbit (follower in leader frame)
    ax = fig.add_subplot(2, 4, 2, projection="3d")
    s = ax.scatter(rel[:, 0], rel[:, 1], rel[:, 2], c=tt, cmap="viridis", s=8)
    ax.plot(rel[:, 0], rel[:, 1], rel[:, 2], color="0.6", lw=0.7)
    ax.scatter([0], [0], [0], marker="*", c="black", s=180, label="leader")
    fig.colorbar(s, ax=ax, pad=0.12, label="t [s]", shrink=0.6)
    ax.set(
        title="Follower in leader-relative frame",
        xlabel="x [m]",
        ylabel="y [m]",
        zlabel="z [m]",
    )
    _equal_3d(ax, np.vstack([rel, np.zeros(3)]))
    ax.legend(loc="upper left", fontsize=7)

    # (3) inter-drone distance vs bounds
    ax = fig.add_subplot(2, 4, 3)
    ax.plot(tt, dist, color="tab:orange", lw=1.5)
    ax.axhline(lo, color="0.5", ls="--", lw=1, label="bounds")
    ax.axhline(hi, color="0.5", ls="--", lw=1)
    ax.fill_between(tt, lo, hi, color="0.85", alpha=0.4)
    ax.set(title="Inter-drone distance", xlabel="t [s]", ylabel="m")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (4) relative attitude (the directly-observed, optimized rotational DOF)
    ax = fig.add_subplot(2, 4, 4)
    ang = np.rad2deg(2 * np.arccos(np.clip(np.abs(rq[:, 3]), 0, 1)))
    for j, (lab, col) in enumerate(
        zip("xyzw", ["tab:red", "tab:green", "tab:blue", "0.4"], strict=True)
    ):
        ax.plot(tt, rq[:, j], lw=1.0, color=col, label=f"q$_{lab}$")
    ax2 = ax.twinx()
    ax2.plot(tt, ang, lw=1.6, color="tab:purple", alpha=0.8)
    ax2.set_ylabel("rotation angle [deg]", color="tab:purple")
    ax.set(
        title="Relative attitude (follower wrt leader)", xlabel="t [s]", ylabel="quat"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower left")

    # (5) plan time
    ax = fig.add_subplot(2, 4, 5)
    ax.plot(tt, walls, color="tab:green", lw=1.2)
    ax.axhline(100, color="0.6", ls=":", lw=1, label="100 ms (10 Hz)")
    ax.set(
        title=f"Per-tick plan time (median {np.median(walls[steady]):.0f} ms)",
        xlabel="t [s]",
        ylabel="ms",
        ylim=(0, max(250, np.percentile(walls[steady], 99) * 1.2)),
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (6) plan-time distribution
    ax = fig.add_subplot(2, 4, 6)
    ax.hist(walls[steady], bins=30, color="tab:green", alpha=0.8)
    ax.axvline(100, color="0.6", ls=":", lw=1, label="100 ms")
    ax.axvline(
        np.median(walls[steady]),
        color="tab:red",
        ls="--",
        lw=1,
        label=f"median {np.median(walls[steady]):.0f} ms",
    )
    ax.set(title="Plan-time distribution (steady state)", xlabel="ms", ylabel="count")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    # (7) observable directions
    ax = fig.add_subplot(2, 4, 7)
    ax.plot(tt, ndir, color="tab:blue", lw=1.5, drawstyle="steps-post")
    ax.axhline(6, color="0.6", ls=":", lw=1, label="full (6/6)")
    ax.set(
        title="Observable directions", xlabel="t [s]", ylabel="count", ylim=(-0.3, 6.5)
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (8) accumulated min-eigenvalue (observability margin)
    ax = fig.add_subplot(2, 4, 8)
    ax.semilogy(tt, np.maximum(mineig, 1e-300), color="tab:purple", lw=1.5)
    ax.set(
        title="Accumulated Gramian min-eigenvalue",
        xlabel="t [s]",
        ylabel="$\\lambda_{min}$",
    )
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = RESULTS / "example_quadrotor.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
