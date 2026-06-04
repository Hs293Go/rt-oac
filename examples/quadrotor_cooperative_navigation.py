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

Config is **Hydra** (conf/quadrotor.yaml + conf/mode/*.yaml); the ``mode`` group selects a
pedagogical arc through finding #8 (override on the CLI, e.g. ``mode=hybrid``):
  * ``mode=open`` (default) = *open-loop planner*: perfect state feedback (the planner sees the
    true relative state), the soft-min orbit, the level-cruise leader -- the pretty orbit;
  * ``mode=estimation`` = **estimation in the loop**: the same pure observability, but the
    controller now acts on the ``ErrorStateEKF`` *estimate* -- the orbit *diverges* (finding #8);
  * ``mode=hybrid`` = an **exploratory, seed-fragile** closed loop: estimation in the loop + a
    tracking/observability balanced cost (covariance+innovation-scheduled when ``mode.schedule``).
    On a lucky seed it can hold NEES near the chi-square band, but across seeds it does **not**
    robustly resolve the carried-estimation coupling (the earlier "resolution" was overfit to one
    seed). The robust, non-diverging baseline is ``reanchor=true`` -- the paper's validation, which
    pins the EKF mean to truth each step (so only the covariance envelope propagates).
Diagnostics: ``reanchor=true`` (mean pinned to truth), ``plan_on_truth=true`` (carry the mean but
plan on truth) -- the three-way trichotomy that isolates the divergence. See the report
(report/improving_beginners_oac.tex) and docs/carried_estimation_plan.md.

Visualization is **rerun**: a live 3D scene plus live time-series (open loop: plan time /
observability / distance; closed loop: error & 3-sigma / NEES / distance / plan time). A
comprehensive multi-panel matplotlib figure is rendered after the run.

``soft=true`` enforces the inter-drone distance band as a smooth penalty + an unconstrained
L-BFGS-B solve instead of the hard SLSQP constraint (faster, fully JIT-able; see
benchmarks/penalty_solver_probe.py). Great for the open-loop orbit (~85 vs ~115 ms, band held)
but NOT for ``mode=hybrid`` (the delicate loop destabilizes under it -- keep hard there).

Open loop:    uv run python examples/quadrotor_cooperative_navigation.py [spawn=true] [soft=true]
Estimation:   uv run python examples/quadrotor_cooperative_navigation.py mode=estimation  (#8 diverges)
Paper valid.: uv run python examples/quadrotor_cooperative_navigation.py mode=estimation reanchor=true
Hybrid (expl):uv run python examples/quadrotor_cooperative_navigation.py mode=hybrid       (seed-fragile)
"""

import dataclasses
import pathlib

from example_lib import math as elmath
from example_lib.models import inter_quadrotor_pose as mdl
from example_lib.visualization import visualization as viz
import hydra
import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
from observability_aware_control import integrator
from omegaconf import DictConfig
import rerun as rr
import rerun.blueprint as rrb
from scipy import stats
import tqdm

import rt_oac  # noqa: F401
from rt_oac import metrics, report, tracking_cost
from rt_oac.balanced_cost import make_balanced
from rt_oac.controller import RTController
from rt_oac.error_state_ekf import ErrorStateEKF
from rt_oac.scenario import build_scenario
from rt_oac.unscented_kf import ManifoldUKF
from rt_oac.warmstart import warm_guess

mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Run config (mode arc, band, horizon, leader speed, soft/spawn/save) lives in conf/ -- see
# conf/quadrotor.yaml + conf/mode/{open,estimation,hybrid}.yaml. The mode group bundles
# feedback/objective/band/horizon (and the velocity-damped standoff anchor for hybrid), so the
# three-mode #8 arc is one key (mode=open|estimation|hybrid) rather than a tangle of flags.
LEADER_ALT = 10.0  # world-frame altitude of the synthesized level-cruise leader
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


@hydra.main(version_base=None, config_path="conf", config_name="quadrotor")
def main(cfg: DictConfig):
    # The mode (conf/mode/*.yaml) bundles feedback/objective/band/horizon; steps falls back to
    # the mode's validated default (open=300; closed-loop=120, where #8 / its resolution show).
    cfg.steps = cfg.steps if cfg.steps is not None else cfg.mode.default_steps
    lo, hi = float(cfg.mode.band[0]), float(cfg.mode.band[1])
    sc = build_scenario(gramian_metric=metrics.neg_softmin_eig)
    if cfg.mode.feedback == "estimate":
        # Closed-loop modes default to the scenario's (near-stationary min-snap) leader -- the
        # config the coupling work was validated on (drone_coupling_eval.py). With that leader the
        # world-frame follower path sits on top of its relative orbit (a near-ego view). Setting
        # world_leader=true swaps in a level constant-velocity cruise so the world-frame scene is
        # genuinely absolute (a moving helix). The relative dynamics see only the leader's hover
        # inputs (f=g, omega=0), so the relative orbit -- and the coupling NEES/error -- are
        # Galilean-invariant w.r.t. the leader's constant velocity (verified).
        if cfg.world_leader:
            n_lead = cfg.steps + sc.window + 1
            x_leader, u_leader = level_cruise_leader(
                cfg.leader_speed, n_lead, sc.cfg["sim"]["integrator_dt"]
            )
            sc = dataclasses.replace(sc, x_leader=x_leader, u_leader=u_leader)
        run_closed_loop(sc, cfg, lo, hi)
        return
    # Open-loop orbit: replace the near-frozen min-snap leader with a level constant-velocity
    # cruise so the world-frame scene covers ground (the relative orbit is Galilean-invariant).
    n_lead = cfg.steps + sc.window + 1
    x_leader, u_leader = level_cruise_leader(
        cfg.leader_speed, n_lead, sc.cfg["sim"]["integrator_dt"]
    )
    sc = dataclasses.replace(sc, x_leader=x_leader, u_leader=u_leader)
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
        constraint_mode="soft" if cfg.soft else "hard",
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
    rr.init("rt_oac_quadrotor", spawn=cfg.spawn)
    rrd = cfg.save or str(RESULTS / "example_quadrotor.rrd")
    if not cfg.spawn:
        rr.save(rrd)
    style_series(lo, hi)
    send_blueprint()

    # Note: set time before initializing lines/frames
    rr.set_time("/time", duration=0)
    # Static world: the recorded leader path (so the live scene has context from t=0).
    lead_path = np.asarray(sc.x_leader[: cfg.steps, 0:3])
    rr.log(
        "/sim/leader/path",
        rr.LineStrips3D(lead_path[None], colors=[90, 90, 90]),
        static=True,
    )
    # Scene helpers: each logs its body-axis Arrows3D once and a per-step Transform3D at the
    # SAME entity path, so the axes (and the body frame) move with the pose; traces are
    # logged in world coordinates as siblings (not double-transformed).
    leader_viz = viz.PoseReferenceFrame("/sim/leader")
    leader_tr = viz.PositionTrace("/sim/leader", max_length=cfg.steps)
    foll_viz = viz.PoseReferenceFrame("/sim/follower")
    foll_tr = viz.PositionTrace("/sim/follower", max_length=cfg.steps)

    x = np.concatenate([[2.0, 0.0, 0.0], quat_z(20), [0.0, 1.0, 0.0]])  # non-MUC start
    rec = {
        k: [] for k in ("rel", "rq", "lead", "foll", "dist", "ndir", "mineig", "walls")
    }
    prev_u = None
    t = 0.0
    for i in tqdm.trange(cfg.steps):
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

    summary = summarize_and_plot(rec, dt, lo, hi, cfg.steps)
    if cfg.report:
        tag = f"quad_{cfg.mode.name}" + ("_soft" if cfg.soft else "")
        arrays = {k: np.array(v) for k, v in rec.items()}
        arrays["t"] = np.arange(cfg.steps) * dt
        report.dump(
            cfg.report,
            tag,
            summary={
                "kind": "quad",
                "variant": cfg.mode.name,
                "soft": bool(cfg.soft),
                "hybrid": False,
                "feedback": cfg.mode.feedback,
                "objective": cfg.mode.objective,
                **summary,
            },
            arrays=arrays,
        )
        print(f"report data -> {cfg.report}/{tag}.{{npz,json}}")
    if not cfg.spawn:
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
    summary = {
        "steps": int(steps),
        "dt": float(dt),
        "band_lo": float(lo),
        "band_hi": float(hi),
        "plan_median_ms": float(np.median(walls[steady])),
        "plan_p95_ms": float(np.percentile(walls[steady], 95)),
        "plan_tick0_ms": float(walls[0]),
        "ndir_median": int(np.median(ndir)),
        "ndir_full": 6,
        "mineig_median": float(np.median(mineig)),
        "dist_min": float(dist.min()),
        "dist_max": float(dist.max()),
    }

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
    return summary


# =============== closed-loop path (--estimation pure obs / --hybrid balanced) =============
def _tangent_error(x_hat, x_true):
    """9-vector tangent error x_hat boxminus x_true (manifold-consistent)."""
    diff = x_hat - x_true
    dtheta = np.array(
        elmath.quaternion_log(
            elmath.quaternion_product(
                elmath.quaternion_inverse(jnp.asarray(x_true[3:7])),
                jnp.asarray(x_hat[3:7]),
            )
        )
    )
    return np.concatenate([diff[0:3], dtheta, diff[7:10]])


_NEES_BLOCKS = {"pos": slice(0, 3), "rot": slice(3, 6), "vel": slice(6, 9)}


def _block_nees(err_t, P):
    """Per-sub-block NEES (pos/rot/vel), each ~ chi^2_3 -- localizes over/under-confidence."""
    out = {}
    for name, sl in _NEES_BLOCKS.items():
        e = err_t[sl]
        out[name] = float(e @ np.linalg.solve(P[sl, sl] + 1e-12 * np.eye(3), e))
    return out


def anchor_alpha(i, a0, hold, anneal):
    """Training-wheels schedule alpha(i): hold at ``a0`` for ``hold`` steps, then linearly
    anneal to 0 over ``anneal`` steps (anneal=0 = sharp release), then 0 (carried). a0=0 = off."""
    if a0 <= 0 or i < hold:
        return a0
    if anneal <= 0 or i >= hold + anneal:
        return 0.0
    return a0 * (1.0 - (i - hold) / anneal)


def run_closed_loop(sc, cfg, lo, hi):
    """Closed-loop ESEKF: the controller acts on the EKF *estimate*, not truth.

    ``mode=estimation`` (objective=softmin, no anchor): pure observability -- the #8 pathology,
    where the orbit diverges under estimate feedback. ``mode=hybrid`` (objective=balanced): the
    balanced cost (observability + velocity-damped standoff anchor from cfg.mode) keeps it bounded
    and the estimator consistent. See PROGRESS.md section 7 and experiments/drone_coupling_eval.py.
    """
    hybrid = cfg.mode.objective == "balanced"
    dt = sc.cfg["sim"]["integrator_dt"]
    x0 = np.concatenate([
        [1.0, 0.0, 0.0],
        quat_z(20),
        [0.0, 1.0, 0.0],
    ])  # at the standoff
    if hybrid:
        ref_us = sc.reference_guess(0)
        s_obs = abs(
            float(sc.cost(jnp.asarray(x0), jnp.asarray(ref_us), sc.stlog_dt).objective)
        )
        track = tracking_cost.quadratic_tracking_cost(
            position_indices=tuple(cfg.mode.pos_idx), w_pos=1.0
        )
        cost = make_balanced(
            sc.cost,
            track,
            scheme="normalized",
            s_track=float(cfg.mode.rho) ** 2 * sc.window,
            s_obs=s_obs,
        )
        p_ref = np.tile(np.asarray(cfg.mode.standoff), (sc.window, 1))
    else:
        cost, p_ref = (
            sc.cost,
            None,
        )  # --estimation: pure observability (no anchor) -> #8
    ctrl = RTController(
        cost,
        stlog_dt=sc.stlog_dt,
        lb=np.array(sc.cfg["optim"]["lb"]),
        ub=np.array(sc.cfg["optim"]["ub"]),
        n_inputs=8,
        follower_indices=(4, 5, 6, 7),
        method="SLSQP",
        maxiter=6,
        constraint=mdl.interrobot_distance_squared,
        constraint_bounds=(lo**2, hi**2),
        constraint_mode="soft" if cfg.soft else "hard",
    )

    range_var, att_var = sc.cfg["noise"]["range_var"], sc.cfg["noise"]["att_var"]
    obs_var = np.concatenate([[range_var], np.full(4, att_var)])
    res_var = np.concatenate([[range_var], np.full(3, att_var)])
    input_var = np.tile([0.05, 0.01, 0.01, 0.01], 2)
    proc_var = np.array([0.02, 0.02, 0.02, 1e-4, 1e-4, 1e-4, 0.05, 0.05, 0.05])
    proc_var[0:3] *= float(cfg.proc_pos_scale)  # M2: inflate the position process noise
    sim = jax.jit(
        integrator.Integrator(
            mdl.dynamics, integrator.Methods.EULER, stepsize=dt, manifold=mdl.MANIFOLD
        )
    )
    filter_cls = ManifoldUKF if str(cfg.filter) == "ukf" else ErrorStateEKF
    ekf = filter_cls(
        mdl.dynamics,
        lambda x: mdl.observation(x),
        mdl.MANIFOLD,
        in_cov=np.diag(input_var),
        obs_cov=np.diag(res_var),
        proc_cov=np.diag(proc_var),
        method=integrator.Methods.EULER,
    )

    mode = "hybrid" if hybrid else "estimation"
    rr.init(f"rt_oac_quadrotor_{mode}", spawn=cfg.spawn)
    rrd = cfg.save or str(RESULTS / f"example_quadrotor_{mode}.rrd")
    if not cfg.spawn:
        rr.save(rrd)
    for path, (name, col) in {
        "/graphs/error/err": ("rel-pos error [m]", [230, 40, 40]),
        "/graphs/error/sig": ("3-sigma [m]", [150, 150, 150]),
        "/graphs/nees/val": ("NEES (9-dof)", [180, 80, 220]),
        "/graphs/nees/exp": ("E[NEES]=9", [150, 150, 150]),
        "/graphs/distance/dist": ("distance [m]", [230, 120, 40]),
        "/graphs/solve/ms": ("plan time [ms]", [40, 180, 80]),
    }.items():
        rr.log(path, rr.SeriesLines(names=name, colors=col, widths=2), static=True)
    rr.send_blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/sim", name="Cooperative quadrotors (ESEKF loop)"
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(
                    origin="/graphs/error", name="Rel-pos error & 3-sigma"
                ),
                rrb.TimeSeriesView(origin="/graphs/nees", name="NEES (consistency)"),
                rrb.TimeSeriesView(
                    origin="/graphs/distance", name="Inter-drone distance"
                ),
                rrb.TimeSeriesView(origin="/graphs/solve", name="Plan time [ms]"),
            ),
            column_shares=[1.5, 1.0],
        )
    )
    leader_viz = viz.PoseReferenceFrame("/sim/leader")
    leader_tr = viz.PositionTrace("/sim/leader", max_length=cfg.steps)
    ft_viz = viz.PoseReferenceFrame("/sim/follower_true")
    ft_tr = viz.PositionTrace("/sim/follower_true", max_length=cfg.steps)
    fe_viz = viz.PoseReferenceFrame("/sim/follower_est")

    rng = np.random.default_rng(cfg.seed)
    x_true = x0.copy()
    P = np.diag([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3, 0.4, 0.4, 0.4])
    delta0 = np.array([0.9, 0.9, 0.4, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0])
    x_hat = np.array(mdl.MANIFOLD.boxplus(jnp.asarray(x_true), jnp.asarray(delta0)))

    rec = {
        k: []
        for k in (
            "rel",
            "lead",
            "ftrue",
            "fest",
            "err",
            "sig",
            "nees",
            "dist",
            "walls",
            "nees_pos",
            "nees_rot",
            "nees_vel",
            "alpha",
        )
    }
    scheduled = hybrid and bool(cfg.mode.schedule)
    reanchor = bool(
        cfg.reanchor
    )  # reset EKF mean to truth each step (paper validation setup)
    plan_on_truth = bool(
        cfg.plan_on_truth
    )  # perfect-feedback OA traj; EKF passive (diagnostic)
    # Annealed "training-wheels" truth-anchor (sim-only): a partial pseudo-measurement of the
    # true relative state, strength alpha(i) (a0 at t=0, held, then annealed to 0 = carried).
    a0_anchor = float(cfg.anchor_alpha0)
    hold_anchor, anneal_anchor = int(cfg.anchor_hold), int(cfg.anchor_anneal)
    # M2 estimator-consistency knobs (truth-free): First-Estimates Jacobian (freeze the EKF
    # linearization point, reset every fej_epoch) + a covariance floor on trace(P[pos]).
    fej, fej_epoch = bool(cfg.fej), int(cfg.fej_epoch)
    p_pos_floor = float(cfg.p_pos_floor)
    x_lin = x_hat.copy()  # FEJ linearization reference
    weight = float(cfg.mode.weight) if hybrid else 1.0
    if scheduled:
        s0 = float(cfg.mode.s0)
        w_min, w_max = float(cfg.mode.w_min), float(cfg.mode.w_max)
        nis0 = float(cfg.mode.nis0)
    prev_u, t, dof = None, 0.0, 9
    prev_nis = 0.0  # range-innovation NIS from the previous step (estimator surprise)
    for i in tqdm.trange(cfg.steps):
        if reanchor:
            x_hat = (
                x_true.copy()
            )  # paper validation: pin the EKF mean to truth (no drift)
        if scheduled:
            # Dual control: spend the observability budget only where the estimator can keep up.
            # Back off on BOTH claimed uncertainty (s = trace(P[pos]), P being the prior step's
            # posterior) AND estimator surprise (prev_nis). Covariance alone misses a
            # confident-wrong divergence (P small, true error large); the innovation catches it.
            s = float(np.trace(P[0:3, 0:3]))
            weight = float(
                np.clip(w_max / (1.0 + s / s0 + prev_nis / nis0), w_min, w_max)
            )
        guess = warm_guess(sc.reference_guess(i), prev_u)
        x_plan = (
            x_true if plan_on_truth else x_hat
        )  # plan on truth (open-loop OA) vs estimate
        res = ctrl.solve(x_plan, guess, p_ref=p_ref, weight=weight)
        prev_u = res.u
        u = res.u[0].copy()
        u[4:] += rng.normal(0, np.sqrt(input_var[4:]))
        if fej and i % fej_epoch == 0:
            x_lin = (
                x_hat.copy()
            )  # reset the FEJ linearization point to the current estimate
        jx_lin = jnp.asarray(x_lin)
        if fej:
            x_hat, P = ekf.predict_fej(
                jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(u), dt, jx_lin
            )
        else:
            x_hat, P = ekf.predict(
                jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(u), dt
            )
        x_hat, P = np.array(x_hat), np.array(P)
        x_true = renorm(sim(jnp.asarray(x_true), jnp.asarray(u))[0])
        y = np.array(mdl.observation(jnp.asarray(x_true))) + rng.normal(
            0, np.sqrt(obs_var)
        )
        # Estimator surprise: range-channel innovation NIS at the prior (before the update).
        nu_range = float(y[0] - np.asarray(mdl.observation(jnp.asarray(x_hat)))[0])
        prev_nis = nu_range**2 / range_var
        if fej:
            x_hat, P = ekf.update_fej(
                jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(y), jx_lin
            )
        else:
            x_hat, P = ekf.update(jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(y))
        x_hat, P = np.array(x_hat), np.array(P)
        # Training wheels: pull the carried estimate a fraction alpha(i) toward truth via a
        # covariance-consistent pseudo-measurement (R = P (1-a)/a => gain a*I). alpha=0 = off.
        alpha = min(anchor_alpha(i, a0_anchor, hold_anchor, anneal_anchor), 0.999)
        if alpha > 0:
            r_anchor = P * (1.0 - alpha) / alpha
            x_hat, P = ekf.update_anchor(
                jnp.asarray(x_hat),
                jnp.asarray(P),
                jnp.asarray(x_true),
                jnp.asarray(r_anchor),
            )
            x_hat, P = np.array(x_hat), np.array(P)
        if p_pos_floor > 0:  # M2: PSD-safe lower bound on trace(P[pos]) -- stay honest
            tr = float(np.trace(P[0:3, 0:3]))
            if tr < p_pos_floor:
                P[0:3, 0:3] += ((p_pos_floor - tr) / 3.0) * np.eye(3)

        err_t = _tangent_error(x_hat, x_true)
        nees = float(err_t @ np.linalg.solve(P + 1e-12 * np.eye(9), err_t))
        bnees = _block_nees(err_t, P)
        err = float(np.linalg.norm(x_hat[0:3] - x_true[0:3]))
        sig = float(3 * np.sqrt(max(np.trace(P[0:3, 0:3]) / 3, 0)))
        dist = float(np.linalg.norm(x_true[0:3]))
        xa_t = np.asarray(mdl.to_absolute_state(sc.x_leader[i], jnp.asarray(x_true)))
        xa_e = np.asarray(mdl.to_absolute_state(sc.x_leader[i], jnp.asarray(x_hat)))
        lead_p, lead_q = (
            np.asarray(sc.x_leader[i, 0:3]),
            np.asarray(sc.x_leader[i, 3:7]),
        )
        ft_p, ft_q = xa_t[10:13], xa_t[13:17]
        fe_p, fe_q = xa_e[10:13], xa_e[13:17]
        for k, v in (
            ("rel", x_true[0:3].copy()),
            ("lead", lead_p.copy()),
            ("ftrue", ft_p.copy()),
            ("fest", fe_p.copy()),
            ("err", err),
            ("sig", sig),
            ("nees", nees),
            ("dist", dist),
            ("walls", res.wall_time * 1e3),
            ("nees_pos", bnees["pos"]),
            ("nees_rot", bnees["rot"]),
            ("nees_vel", bnees["vel"]),
            ("alpha", alpha),
        ):
            rec[k].append(v)

        rr.set_time("/time", duration=t)
        leader_viz.set_pose(lead_p, rr.Quaternion(xyzw=lead_q))
        leader_tr.add_position(lead_p)
        ft_viz.set_pose(ft_p, rr.Quaternion(xyzw=ft_q))
        ft_tr.add_position(ft_p)
        fe_viz.set_pose(fe_p, rr.Quaternion(xyzw=fe_q))
        rr.log(
            "/sim/range",
            rr.LineStrips3D(np.array([lead_p, ft_p])[None], colors=[120, 120, 120]),
        )
        rr.log("/graphs/error/err", rr.Scalars(err))
        rr.log("/graphs/error/sig", rr.Scalars(sig))
        rr.log("/graphs/nees/val", rr.Scalars(nees))
        rr.log("/graphs/nees/exp", rr.Scalars(float(dof)))
        rr.log("/graphs/distance/dist", rr.Scalars(dist))
        rr.log("/graphs/solve/ms", rr.Scalars(res.wall_time * 1e3))
        t += dt

    out = {k: np.array(v) for k, v in rec.items()}
    warmup = min(20, cfg.steps // 4)
    label = (
        "HYBRID (balanced cost)" if hybrid else "ESTIMATION (pure obs, #8 pathology)"
    )
    print(f"=== RT-OAC quadrotor {label} -- ESEKF, controller on estimate ===")
    print(
        f"steps {cfg.steps} | rel-pos error final {out['err'][-1]:.2f} m, max {out['err'].max():.2f} m"
        f" | median plan {np.median(out['walls'][1:]):.0f} ms"
    )
    print(
        f"NEES median (after step {warmup}) {np.median(out['nees'][warmup:]):.1f} (E=9) | "
        f"distance [{out['dist'].min():.2f}, {out['dist'].max():.2f}] m (band [{lo:.1f}, {hi:.1f}])"
    )
    print(
        f"block NEES (after {warmup}) pos {np.median(out['nees_pos'][warmup:]):.1f} "
        f"rot {np.median(out['nees_rot'][warmup:]):.1f} vel "
        f"{np.median(out['nees_vel'][warmup:]):.1f} (E=3 each)"
    )
    # M1 discriminator: health over the wheels-OFF steps (alpha==0, after warmup). For a pure
    # carried run this is the whole run; for an annealed run it is the post-release segment.
    off = (out["alpha"] == 0) & (np.arange(cfg.steps) >= warmup)
    if off.any():
        print(
            f"wheels-off (alpha=0, frac {float((out['alpha'] == 0).mean()):.2f}): "
            f"NEES median {np.median(out['nees'][off]):.1f} err max {out['err'][off].max():.2f} m"
        )
    standoff = np.asarray(cfg.mode.standoff[0:3]) if hybrid else None
    plot_hybrid(out, dt, lo, hi, cfg.steps, mode, hybrid, standoff)
    if cfg.report:
        diag = "_reanchor" if reanchor else ("_planontruth" if plan_on_truth else "")
        tag = f"quad_{mode}{diag}" + ("_soft" if cfg.soft else "")
        arrays = {**out, "t": np.arange(cfg.steps) * dt}
        report.dump(
            cfg.report,
            tag,
            summary={
                "kind": "quad",
                "variant": mode,
                "soft": bool(cfg.soft),
                "hybrid": hybrid,
                "reanchor": reanchor,
                "plan_on_truth": plan_on_truth,
                "anchor_alpha0": a0_anchor,
                "anchor_hold": hold_anchor,
                "anchor_anneal": anneal_anchor,
                "feedback": "estimate",
                "objective": "balanced" if hybrid else "softmin",
                "steps": int(cfg.steps),
                "dt": float(dt),
                "band_lo": float(lo),
                "band_hi": float(hi),
                "nees_dof": dof,
                "err_final_m": float(out["err"][-1]),
                "err_max_m": float(out["err"].max()),
                "err_rmse_m": float(np.sqrt((out["err"] ** 2).mean())),
                "plan_median_ms": float(np.median(out["walls"][1:])),
                "nees_median": float(np.median(out["nees"][warmup:])),
                "nees_pos_median": float(np.median(out["nees_pos"][warmup:])),
                "nees_rot_median": float(np.median(out["nees_rot"][warmup:])),
                "nees_vel_median": float(np.median(out["nees_vel"][warmup:])),
                "in3s": float(np.mean(out["err"] <= out["sig"])),
                "dist_min": float(out["dist"].min()),
                "dist_max": float(out["dist"].max()),
                "diverged": bool(out["err"].max() > 5.0 or out["dist"].max() > 5.0),
            },
            arrays=arrays,
        )
        print(f"report data -> {cfg.report}/{tag}.{{npz,json}}")
    if not cfg.spawn:
        print(f"rerun recording written; open with:  rerun {rrd}")


def plot_hybrid(rec, dt, lo, hi, steps, mode, hybrid, standoff):
    tt = np.arange(steps) * dt
    dof = 9
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        (
            "RT-OAC quadrotor HYBRID: balanced cost (obs + velocity-damped standoff) + ESEKF "
            "-- the #8-resolved closed loop"
            if hybrid
            else "RT-OAC quadrotor ESTIMATION: pure observability + ESEKF estimate feedback "
            "-- the finding-#8 divergence (no anchor)"
        ),
        fontsize=13,
    )
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    ax.plot(
        rec["lead"][:, 0],
        rec["lead"][:, 1],
        rec["lead"][:, 2],
        color="black",
        lw=2,
        label="leader",
    )
    ax.plot(
        rec["ftrue"][:, 0],
        rec["ftrue"][:, 1],
        rec["ftrue"][:, 2],
        color="tab:red",
        lw=1.4,
        label="follower true",
    )
    ax.plot(
        rec["fest"][:, 0],
        rec["fest"][:, 1],
        rec["fest"][:, 2],
        color="tab:orange",
        lw=1.0,
        ls="--",
        label="follower est",
    )
    _equal_3d(ax, np.vstack([rec["lead"], rec["ftrue"]]))
    ax.set(
        title="World-frame trajectory", xlabel="x [m]", ylabel="y [m]", zlabel="z [m]"
    )
    ax.legend(fontsize=7)

    ax = fig.add_subplot(2, 3, 2, projection="3d")
    s = ax.scatter(
        rec["rel"][:, 0], rec["rel"][:, 1], rec["rel"][:, 2], c=tt, cmap="viridis", s=8
    )
    extra = rec["rel"]
    if (
        standoff is not None
    ):  # hybrid: mark the standoff set-point (none in pure-obs estimation)
        ax.scatter(*standoff, marker="*", c="black", s=160, label="standoff")
        extra = np.vstack([rec["rel"], standoff])
    fig.colorbar(s, ax=ax, pad=0.12, label="t [s]", shrink=0.6)
    _equal_3d(ax, extra)
    ax.set(
        title="Follower in leader-relative frame",
        xlabel="x [m]",
        ylabel="y [m]",
        zlabel="z [m]",
    )
    ax.legend(fontsize=7)

    ax = fig.add_subplot(2, 3, 3)
    ax.plot(tt, rec["err"], color="tab:red", lw=2, label="rel-pos error")
    ax.plot(tt, rec["sig"], color="0.5", lw=1.5, label="3-sigma")
    ax.fill_between(tt, 0, rec["sig"], color="0.85", alpha=0.5)
    ax.set(title="Estimation error vs 3-sigma", xlabel="t [s]", ylabel="m")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 4)
    ax.plot(tt, rec["nees"], color="tab:purple", lw=1.3)
    ax.axhline(dof, color="k", ls=":", lw=1, label="E[NEES]=9")
    ax.axhspan(
        stats.chi2.ppf(0.025, dof),
        stats.chi2.ppf(0.975, dof),
        color="0.85",
        alpha=0.6,
        label="chi2 95%",
    )
    ax.set(
        title="NEES (estimator consistency)",
        xlabel="t [s]",
        ylabel="NEES",
        ylim=(0, min(rec["nees"].max() * 1.1 + 1, 200)),
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 5)
    ax.plot(tt, rec["dist"], color="tab:blue", lw=1.5)
    ax.axhline(lo, color="0.5", ls="--", lw=1, label="band")
    ax.axhline(hi, color="0.5", ls="--", lw=1)
    ax.set(title="Inter-drone distance", xlabel="t [s]", ylabel="m")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 6)
    ax.plot(tt[1:], rec["walls"][1:], color="tab:green", lw=1.2)
    ax.axhline(
        np.median(rec["walls"][1:]),
        color="0.5",
        ls=":",
        lw=1,
        label=f"median {np.median(rec['walls'][1:]):.0f} ms",
    )
    ax.set(title="Per-tick plan time", xlabel="t [s]", ylabel="ms", ylim=(0, None))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = RESULTS / f"example_quadrotor_{mode}.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
