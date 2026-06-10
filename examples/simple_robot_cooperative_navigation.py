"""Headline example: real-time planar leader-follower cooperative navigation, with rerun.

**Rung 1** of the bottom-up RANGE-ONLY OAC ladder (planar unicycle -> 3D point-mass -> flat-output
``[x,y,z,psi]``, see ``examples/flat_robot_cooperative_navigation.py`` and PROGRESS section 9).

The future-work counterpart of the companion repo's
``examples/simple_robot_cooperative_navigation.py``. Same planar unicycle leader-follower
(``leader_follower_robots``; the follower's position is observable only through the
inter-robot range), with a **carried** EKF estimate (the controller sees only the estimate,
truth evolves separately, error accumulates) -- unlike the companion example, which
re-anchored the estimate to truth each step.

Frontier method: **log-det** objective + **SLSQP** early-stopped at 6 iters. Compares
observability-aware control against a no-OAC follower (drives straight) on closed-loop EKF
estimation accuracy, from a large initial follower-position error. Measured result:
OAC drives the unobservable initial error down (~5x lower final error, 0.79 -> 0.16 m;
~3.3x lower RMSE, 0.80 -> 0.24 m) where no-OAC cannot, at ~3 ms/solve. See ``PROGRESS.md``
/ ``results/RESULTS.md``.

Visualization is **rerun**, mirroring the companion repo's examples: a live 2D scene of the
leader, the true follower, the EKF estimate and its shrinking 3-sigma covariance ellipse,
plus live time-series of the OAC-vs-no-OAC estimation error, the accumulated observability,
the solve time, and the inter-robot distance. A comprehensive multi-panel matplotlib figure
is rendered after the run.

Config is **Hydra** (conf/planar.yaml). ``hybrid=true`` swaps the log-det OAC for the balanced
cost (observability + formation standoff, PROGRESS section 7); ``soft=true`` enforces the
inter-robot distance band as a smooth penalty + unconstrained L-BFGS-B instead of the hard
SLSQP constraint (faster, JIT-able). Both work here (and compose), unlike the quadrotor hybrid,
which needs the hard constraint. (The planar is inherently estimation-in-the-loop: a carried
EKF is its defining feature, so there is no feedback mode group.)

Live viewer:  uv run python examples/simple_robot_cooperative_navigation.py spawn=true
Headless:     uv run python examples/simple_robot_cooperative_navigation.py [hybrid=true] [soft=true]
"""

import functools
import pathlib

from example_lib.misc import simple_ekf
from example_lib.models import leader_follower_robots as mdl
from example_lib.visualization import visualization as viz
import hydra
import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
from observability_aware_control import integrator, observability_cost
from omegaconf import DictConfig
import rerun as rr
import rerun.blueprint as rrb
import tqdm

import rt_oac  # noqa: F401
from rt_oac import metrics, report, tracking_cost
from rt_oac.balanced_cost import make_balanced
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
DIST_BOUNDS = (0.2, 6.0)
# --- hybrid (--hybrid) mode: tracking/observability balance (PROGRESS section 7) ---
# The follower tracks a moving formation standoff (leader prediction + offset) balanced with
# observability via NormalizedWeightedSum. The unicycle has no relative-velocity state to damp,
# so the anchor is position-only (indices 3,4). The estimator is already closed-loop here.
FORMATION_OFFSET = X0[3:5] - X0[0:2]  # nominal follower-minus-leader offset = [-0.5, 5]
HYBRID_POS_IDX = (3, 4)
HYBRID_RHO = 1.0  # tracking scale (s_track = RHO^2 * window)
HYBRID_WEIGHT = 1.0  # dimensionless observability weight in the normalized balance
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"


def leader_pred(leader_state):
    """(window, 2) predicted leader positions over the horizon (rolls LEADER_U)."""
    px, py, th = leader_state
    ks = np.arange(WINDOW)
    v = LEADER_U[0]
    return np.stack(
        [px + ks * DT * v * np.cos(th), py + ks * DT * v * np.sin(th)], axis=1
    )


PureYaw = functools.partial(rr.RotationAxisAngle, axis=[0, 0, 1])


def build_controller(soft=False):
    cost = observability_cost.ObservabilityCost(
        mdl.dynamics,
        mdl.observation,
        DT,
        gramian_kw={"order": 1, "var": VAR},
        gramian_metric=metrics.neg_logdet,
        observed_indices=(),
    )
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
        constraint_bounds=DIST_BOUNDS,
        constraint_mode="soft" if soft else "hard",
    )
    return ctrl, cost


def build_hybrid_controller(soft=False):
    """Balanced cost (normalized observability + formation standoff anchor)."""
    cost = observability_cost.ObservabilityCost(
        mdl.dynamics,
        mdl.observation,
        DT,
        gramian_kw={"order": 1, "var": VAR},
        gramian_metric=metrics.neg_logdet,
        observed_indices=(),
    )
    ref_us = np.tile(np.r_[LEADER_U, 1.0, 0.0], (WINDOW, 1))
    s_obs = abs(float(cost(jnp.asarray(X0), jnp.asarray(ref_us), STLOG_DT).objective))
    track = tracking_cost.quadratic_tracking_cost(
        position_indices=HYBRID_POS_IDX, w_pos=1.0
    )
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
        constraint_mode="soft" if soft else "hard",
    )
    return ctrl, cost


def cov_ellipse(center2, P2, nsig=3.0, n=48):
    """3-sigma covariance ellipse as (n,3) points in the z=0 plane."""
    vals, vecs = np.linalg.eigh(0.5 * (P2 + P2.T))
    vals = np.clip(vals, 0.0, None)
    t = np.linspace(0, 2 * np.pi, n)
    circ = np.stack([np.cos(t), np.sin(t)])
    pts = (vecs @ (nsig * np.sqrt(vals)[:, None] * circ)).T + np.asarray(center2)
    return np.column_stack([pts, np.zeros(n)])


def _acc_min_eig(gram, x, u):
    g = np.asarray(gram(jnp.asarray(x), u))
    acc = 0.5 * (g + np.swapaxes(g, -1, -2)).sum(0)
    return float(np.linalg.eigvalsh(acc)[0])


def run(use_oac, ctrl, gram, *, stream=False, noac_err=None, hybrid=False):
    """One closed-loop run. If ``stream``, log it live to rerun (overlaying ``noac_err``)."""
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

    if stream:
        leader = viz.PoseReferenceFrame("/sim/leader")
        leader_tr = viz.PositionTrace("/sim/leader", max_length=STEPS)
        ftrue = viz.PoseReferenceFrame("/sim/follower_true")
        ftrue_tr = viz.PositionTrace("/sim/follower_true", max_length=STEPS)
        fest = viz.PoseReferenceFrame("/sim/follower_est")

    rec = {
        k: []
        for k in (
            "lead",
            "ftrue",
            "fest",
            "Pf",
            "err",
            "sig",
            "mineig",
            "dist",
            "walls",
        )
    }
    prev_u = None
    t = 0.0
    for i in tqdm.trange(STEPS):
        if use_oac:
            guess = np.tile(np.r_[LEADER_U, 1.0, 0.0], (WINDOW, 1))
            if prev_u is not None:
                guess[:, 2:4] = np.vstack([prev_u[1:, 2:4], prev_u[-1:, 2:4]])
            if hybrid:  # balanced cost: thread the moving formation standoff + weight
                p_ref = leader_pred(x_hat[0:3]) + FORMATION_OFFSET
                res = ctrl.solve(x_hat, guess, p_ref=p_ref, weight=HYBRID_WEIGHT)
            else:
                res = ctrl.solve(x_hat, guess)  # controller acts on the ESTIMATE
            prev_u = res.u
            u_foll = res.u[0, 2:4]
            rec["walls"].append(res.wall_time * 1e3)
            rec["mineig"].append(_acc_min_eig(gram, x_hat, res.u))
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

        err = float(np.linalg.norm(x_hat[3:5] - x_true[3:5]))
        # 3-sigma envelope for the position-error NORM = semi-major axis of the 3-sigma
        # covariance ellipse (largest eigenvalue), matching the ellipse drawn by cov_ellipse.
        sig = float(3 * np.sqrt(max(np.linalg.eigvalsh(P[3:5, 3:5])[-1], 0.0)))
        dist = float(np.linalg.norm(x_true[0:2] - x_true[3:5]))
        rec["lead"].append(x_true[0:2].copy())
        rec["ftrue"].append(x_true[3:5].copy())
        rec["fest"].append(x_hat[3:5].copy())
        rec["Pf"].append(P[3:5, 3:5].copy())
        rec["err"].append(err)
        rec["sig"].append(sig)
        rec["dist"].append(dist)
        t += DT

        if stream:
            rr.set_time("/time", duration=t)
            leader.set_pose(np.append(x_true[0:2], 0.0), PureYaw(angle=x_true[2]))
            leader_tr.add_position(np.append(x_true[0:2], 0.0))
            ftrue.set_pose(np.append(x_true[3:5], 0.0), PureYaw(angle=x_true[5]))
            ftrue_tr.add_position(np.append(x_true[3:5], 0.0))
            fest.set_pose(np.append(x_hat[3:5], 0.0), PureYaw(angle=x_hat[5]))
            rr.log(
                "/sim/follower_est/cov3sigma",
                rr.LineStrips3D(
                    cov_ellipse(x_hat[3:5], P[3:5, 3:5])[None], colors=[230, 160, 40]
                ),
            )
            rr.log(
                "/sim/range",
                rr.LineStrips3D(
                    np.array([
                        np.append(x_true[0:2], 0.0),
                        np.append(x_true[3:5], 0.0),
                    ])[None],
                    colors=[120, 120, 120],
                ),
            )
            rr.log("/graphs/error/oac", rr.Scalars(err))
            rr.log("/graphs/error/three_sigma", rr.Scalars(sig))
            if noac_err is not None:
                rr.log("/graphs/error/noac", rr.Scalars(float(noac_err[i])))
            rr.log("/graphs/observability/min_eig", rr.Scalars(rec["mineig"][-1]))
            rr.log("/graphs/solve/ms", rr.Scalars(rec["walls"][-1]))
            rr.log("/graphs/distance/dist", rr.Scalars(dist))
            rr.log("/graphs/distance/min", rr.Scalars(DIST_BOUNDS[0]))
            rr.log("/graphs/distance/max", rr.Scalars(DIST_BOUNDS[1]))

    out = {k: np.array(v) for k, v in rec.items() if v}
    out["rmse"] = float(np.sqrt((out["err"] ** 2).mean()))
    out["final"] = float(out["err"][-1])
    return out


def style_series():
    spec = {
        "/graphs/error/oac": ("OAC error [m]", [230, 40, 40]),
        "/graphs/error/noac": ("no-OAC error [m]", [150, 150, 150]),
        "/graphs/error/three_sigma": ("OAC 3-sigma [m]", [230, 160, 40]),
        "/graphs/observability/min_eig": ("accumulated min-eig", [40, 120, 230]),
        "/graphs/solve/ms": ("solve time [ms]", [40, 180, 80]),
        "/graphs/distance/dist": ("inter-robot distance [m]", [180, 80, 220]),
        "/graphs/distance/min": ("min bound", [150, 150, 150]),
        "/graphs/distance/max": ("max bound", [150, 150, 150]),
    }
    for path, (name, color) in spec.items():
        rr.log(path, rr.SeriesLines(names=name, colors=color, widths=2), static=True)


def send_blueprint():
    rr.send_blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/sim", name="Leader-follower under OAC"),
            rrb.Vertical(
                rrb.TimeSeriesView(
                    origin="/graphs/error", name="Follower-position error & 3-sigma"
                ),
                rrb.TimeSeriesView(
                    origin="/graphs/observability", name="Accumulated observability"
                ),
                rrb.TimeSeriesView(origin="/graphs/solve", name="Solve time [ms]"),
                rrb.TimeSeriesView(
                    origin="/graphs/distance", name="Inter-robot distance [m]"
                ),
                row_shares=[1, 1, 1, 1],
            ),
            column_shares=[1.4, 1.0],
        )
    )


@hydra.main(version_base=None, config_path="conf", config_name="planar")
def main(cfg: DictConfig):
    ctrl, cost = (
        build_hybrid_controller(cfg.soft) if cfg.hybrid else build_controller(cfg.soft)
    )
    gram = jax.jit(
        lambda x, u: cost(x, jnp.asarray(u), STLOG_DT, return_gramians=True).gramians
    )

    rr.init("rt_oac_planar", spawn=cfg.spawn)
    suffix = "_hybrid" if cfg.hybrid else ""
    rrd = cfg.save or str(RESULTS / f"example_planar{suffix}.rrd")
    if not cfg.spawn:
        rr.save(rrd)
    style_series()
    send_blueprint()

    # no-OAC first (cheap, no solve) so its error can be overlaid on the live OAC run
    noac = run(False, ctrl, gram, stream=False)
    oac = run(True, ctrl, gram, stream=True, noac_err=noac["err"], hybrid=cfg.hybrid)

    summarize_and_plot(oac, noac, hybrid=cfg.hybrid)
    if cfg.report:
        variant = "hybrid" if cfg.hybrid else "logdet"
        tag = "planar_" + variant + ("_soft" if cfg.soft else "")
        arrays = {
            f"oac_{k}": np.asarray(v)
            for k, v in oac.items()
            if k not in {"rmse", "final"}
        }
        arrays.update({
            f"noac_{k}": np.asarray(v)
            for k, v in noac.items()
            if k not in {"rmse", "final"}
        })
        arrays["t"] = np.arange(STEPS) * DT
        report.dump(
            cfg.report,
            tag,
            summary={
                "kind": "planar",
                "variant": variant,
                "soft": bool(cfg.soft),
                "hybrid": bool(cfg.hybrid),
                "feedback": "estimate",
                "objective": "balanced" if cfg.hybrid else "logdet",
                "steps": int(STEPS),
                "dt": float(DT),
                "band_lo": float(DIST_BOUNDS[0]),
                "band_hi": float(DIST_BOUNDS[1]),
                "oac_rmse_m": float(oac["rmse"]),
                "oac_final_m": float(oac["final"]),
                "noac_rmse_m": float(noac["rmse"]),
                "noac_final_m": float(noac["final"]),
                "plan_median_ms": float(np.median(oac["walls"][1:])),
                "plan_tick0_ms": float(oac["walls"][0]),
                "mineig_median": float(np.median(oac["mineig"])),
                "dist_min": float(oac["dist"].min()),
                "dist_max": float(oac["dist"].max()),
            },
            arrays=arrays,
        )
        print(f"report data -> {cfg.report}/{tag}.{{npz,json}}")
    if not cfg.spawn:
        print(f"rerun recording written; open with:  rerun {rrd}")


def summarize_and_plot(oac, noac, hybrid=False):
    tt = np.arange(STEPS) * DT
    mode = "HYBRID (balanced obs+standoff)" if hybrid else "frontier OPC"
    print(f"=== RT-OAC planar cooperative navigation ({mode} + carried EKF) ===")
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

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        "RT-OAC planar cooperative navigation: log-det + SLSQP@6 + carried EKF "
        "(OAC vs no-OAC)",
        fontsize=14,
    )

    # (1) XY trajectories with periodic 3-sigma ellipses
    ax = fig.add_subplot(2, 3, 1)
    ax.plot(noac["lead"][:, 0], noac["lead"][:, 1], color="black", lw=2, label="leader")
    ax.plot(
        oac["ftrue"][:, 0],
        oac["ftrue"][:, 1],
        color="tab:red",
        lw=1.6,
        label="follower true (OAC)",
    )
    ax.plot(
        oac["fest"][:, 0],
        oac["fest"][:, 1],
        color="tab:orange",
        lw=1.2,
        ls="--",
        label="follower est (OAC)",
    )
    ax.plot(
        noac["ftrue"][:, 0],
        noac["ftrue"][:, 1],
        color="tab:gray",
        lw=1.4,
        ls=":",
        label="follower true (no-OAC)",
    )
    for k in range(0, STEPS, max(1, STEPS // 8)):
        ell = cov_ellipse(oac["fest"][k], oac["Pf"][k])
        ax.plot(ell[:, 0], ell[:, 1], color="tab:orange", lw=0.7, alpha=0.5)
    ax.set(title="Trajectories (XY)", xlabel="x [m]", ylabel="y [m]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")

    # (2) follower-position error OAC vs no-OAC
    ax = fig.add_subplot(2, 3, 2)
    ax.plot(tt, noac["err"], color="tab:gray", lw=2, label="no-OAC error")
    ax.plot(tt, oac["err"], color="tab:red", lw=2, label="OAC error")
    ax.plot(
        tt,
        oac["sig"],
        color="tab:orange",
        lw=1,
        ls="--",
        alpha=0.7,
        label="OAC 3-sigma",
    )
    ax.set(title="Follower-position estimation error", xlabel="t [s]", ylabel="m")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (3) RMSE / final-error bars
    ax = fig.add_subplot(2, 3, 3)
    xb = np.arange(2)
    ax.bar(
        xb - 0.18,
        [noac["rmse"], oac["rmse"]],
        0.36,
        color=["tab:gray", "tab:red"],
        label="RMSE",
    )
    ax.bar(
        xb + 0.18,
        [noac["final"], oac["final"]],
        0.36,
        color=["0.7", "tab:orange"],
        label="final",
    )
    ax.set_xticks(xb, ["no-OAC", "OAC"])
    ax.set(title="Estimation error summary", ylabel="m")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    # (4) accumulated observability (min-eig)
    ax = fig.add_subplot(2, 3, 4)
    ax.plot(tt, oac["mineig"], color="tab:blue", lw=1.5)
    ax.set(
        title="Accumulated Gramian min-eigenvalue (OAC)",
        xlabel="t [s]",
        ylabel="$\\lambda_{min}$",
    )
    ax.grid(alpha=0.3)

    # (5) solve time (drop tick-0 compile so the steady-state ms is legible)
    ax = fig.add_subplot(2, 3, 5)
    med = np.median(oac["walls"][1:])
    ax.plot(tt[1:], oac["walls"][1:], color="tab:green", lw=1.2)
    ax.axhline(med, color="0.5", ls=":", lw=1, label=f"median {med:.1f} ms")
    ax.set(
        title=f"Per-tick solve time (OAC), tick-0 compile {oac['walls'][0]:.0f} ms",
        xlabel="t [s]",
        ylabel="ms",
        ylim=(0, max(np.percentile(oac["walls"][1:], 99) * 1.4, med * 3)),
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (6) inter-robot distance vs bounds
    ax = fig.add_subplot(2, 3, 6)
    ax.plot(tt, oac["dist"], color="tab:purple", lw=1.5, label="OAC")
    ax.plot(tt, noac["dist"], color="tab:gray", lw=1.2, ls=":", label="no-OAC")
    ax.axhline(DIST_BOUNDS[0], color="0.5", ls="--", lw=1)
    ax.axhline(DIST_BOUNDS[1], color="0.5", ls="--", lw=1, label="bounds")
    ax.set(title="Inter-robot distance", xlabel="t [s]", ylabel="m")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = RESULTS / (f"example_planar{'_hybrid' if hybrid else ''}.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
