"""Headline example: real-time 3D FLAT-OUTPUT cooperative navigation, with rerun (rung 3a).

The top of the bottom-up range-only OAC ladder: planar unicycle (rung 1,
``simple_robot_cooperative_navigation.py``) -> 3D point-mass -> here, the quadrotor FLAT-OUTPUT space
``sigma = [x, y, z, psi]``. By differential flatness (Mellinger-Kumar) any smooth ``sigma(t)`` is
quad-realizable, so this is the principled OA-planning model: the smallest state that is the quad's
genuine planning space, with the attitude dynamics delegated to a subservient geometric tracker (see
``experiments/flatout_bridge.py`` for the quad-realizability bridge).

Two flat-output robots (leader + follower), world-frame velocity kinematics ``xdot = u``. **RANGE-ONLY**
cooperative localization: the leader pose is known (GNSS) and both yaws are measured (IMU), but the
**follower's position is observable only through the inter-robot range** -- so its two tangential
(cross-range) directions are observable *only under relative motion*. The controller acts on a **carried**
EKF estimate (truth evolves separately, error accumulates). yaw is OA-inert in a range-only world (a free
flat output), so the OA lever is purely translation.

Frontier method (the rung-2 stable recipe): **soft-min-eigenvalue STLOG** (E-optimality, tames the
log-det baseline-maximizing drive-away) + a bounded inter-robot distance + a symmetry-breaking first
guess (the straight-line start is a degenerate critical point of the observability objective) +
**SLSQP** early-stopped at 6 iters. Compares OA against a no-OAC follower (flies straight alongside the
leader) on closed-loop EKF accuracy, from a large initial follower-position error in the *unobservable*
(tangential) direction. OA drives that error down where no-OAC cannot, at a few ms/solve.

Visualization is **rerun**: a live 3D scene (leader, true follower, EKF estimate + its shrinking
3-sigma covariance ellipsoid, the range link) plus live time-series (OA-vs-no-OAC error & 3-sigma,
accumulated observability, solve time, inter-robot distance). A multi-panel matplotlib figure follows.

Live viewer:  uv run python examples/flat_robot_cooperative_navigation.py spawn=true
Headless:     uv run python examples/flat_robot_cooperative_navigation.py [soft=true]
"""

import pathlib

from example_lib.misc import simple_ekf
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
from rt_oac import metrics, report
from rt_oac.controller import RTController

mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

DT, STLOG_DT, WINDOW, STEPS, ORDER = 0.1, 0.2, 5, 120, 2
VAR = np.r_[np.full(3, 1e-1), 1e-2, 1e-2, 1e-2]  # [leader pos(3), psi_l, psi_f, range]
INPUT_VAR = np.tile([1e-2, 1e-2, 1e-2, 1e-3], 2)  # [vx,vy,vz, yaw-rate] x 2 agents
X0 = np.array([
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    5.0,
    0.0,
    0.0,
])  # leader [0,0,0,0], follower [0,5,0,0]
LEADER_U = np.array([1.0, 0.0, 0.0, 0.0])  # leader cruises +x
DIST_BOUNDS = (4.5, 5.5)  # bounded standoff (the rung-2 stable recipe; ~5 m formation)
P0 = np.diag([
    1e-3,
    1e-3,
    1e-3,
    1e-3,
    2.0,
    2.0,
    2.0,
    1e-2,
])  # large only on follower position
FB_LB, FB_UB = np.array([-1.0, -2.0, -2.0, -2.0]), np.array([3.0, 2.0, 2.0, 2.0])
INIT_FOLLOWER_OFFSET = np.array([
    2.26,
    0.0,
    0.0,
])  # tangential (unobservable) initial position error
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"


# --- flat-output [x,y,z,psi] point-pair model (world-frame velocity kinematics) ---
def dynamics(x, u):
    return jnp.asarray(u)  # xdot = u


def observation(x, u=None):
    lp, psi_l, fp, psi_f = x[0:3], x[3], x[4:7], x[7]
    return jnp.array([lp[0], lp[1], lp[2], psi_l, psi_f, jnp.linalg.norm(fp - lp)])


def interrobot_distance(x0, us, cost):
    xs, _ = cost.eval_integrator(x0, us)
    return jnp.linalg.norm(xs[:, 4:7] - xs[:, 0:3], axis=1)


def build_controller(soft=False):
    cost = observability_cost.ObservabilityCost(
        dynamics,
        observation,
        DT,
        gramian_kw={"order": ORDER, "var": VAR},
        gramian_metric=metrics.neg_softmin_eig,
        observed_indices=(),
    )
    ctrl = RTController(
        cost,
        stlog_dt=STLOG_DT,
        lb=FB_LB,
        ub=FB_UB,
        n_inputs=8,
        follower_indices=(4, 5, 6, 7),
        method="SLSQP",
        maxiter=6,
        constraint=interrobot_distance,
        constraint_bounds=DIST_BOUNDS,
        constraint_mode="soft" if soft else "hard",
    )
    return ctrl, cost


def cov_ellipsoid(center3, P3, nsig=3.0, n=24):
    """3-sigma covariance ellipsoid as 3 orthogonal principal-plane rings (each (n,3))."""
    vals, vecs = np.linalg.eigh(0.5 * (P3 + P3.T))
    vals = np.clip(vals, 0.0, None)
    axes = vecs * (nsig * np.sqrt(vals))  # columns = scaled principal axes
    t = np.linspace(0, 2 * np.pi, n)
    rings = []
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        ring = (
            axes[:, i][None].T * np.cos(t) + axes[:, j][None].T * np.sin(t)
        ).T + center3
        rings.append(ring)
    return rings


def _acc_min_eig(gram, x, u):
    g = np.asarray(gram(jnp.asarray(x), u))
    acc = 0.5 * (g + np.swapaxes(g, -1, -2)).sum(0)
    return float(np.linalg.eigvalsh(acc)[0])


def run(use_oac, ctrl, gram, *, stream=False, noac_err=None):
    """One closed-loop run (carried EKF). If ``stream``, log live to rerun overlaying ``noac_err``."""
    rng = np.random.default_rng(0)
    sim = jax.jit(integrator.Integrator(dynamics, integrator.Methods.RK4, stepsize=DT))
    ekf = simple_ekf.SimpleEKF(
        lambda x, u, dt: x + dt * dynamics(x, u),
        observation,
        in_cov=np.diag(INPUT_VAR),
        obs_cov=np.diag(VAR),
    )
    x_true = X0.copy()
    P = P0.copy()
    x_hat = x_true + np.r_[0, 0, 0, 0, INIT_FOLLOWER_OFFSET, 0.0]
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
    prev_u, t = None, 0.0
    for i in tqdm.trange(STEPS):
        if use_oac:
            guess = np.tile(np.r_[LEADER_U, LEADER_U], (WINDOW, 1))
            if prev_u is not None:
                guess[:, 4:8] = np.vstack([prev_u[1:, 4:8], prev_u[-1:, 4:8]])
            else:  # symmetry-break the degenerate straight-line critical point of the OA objective
                kk = np.arange(WINDOW)
                guess[:, 4:7] += 0.3 * np.c_[np.sin(kk), np.cos(kk), np.sin(kk + 2.0)]
            res = ctrl.solve(
                jnp.asarray(x_hat), jnp.asarray(guess)
            )  # acts on the ESTIMATE
            prev_u = res.u
            u_foll = np.asarray(res.u[0, 4:8])
            rec["walls"].append(res.wall_time * 1e3)
            rec["mineig"].append(_acc_min_eig(gram, x_hat, res.u))
        else:
            u_foll = (
                LEADER_U.copy()
            )  # drive straight alongside (no observability seeking)
        u = np.r_[LEADER_U, u_foll]
        u_applied = u + np.r_[np.zeros(4), rng.normal(0, np.sqrt(INPUT_VAR[4:]))]
        x_hat, P = ekf.predict(
            jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(u_applied), DT
        )
        x_true = np.array(sim(jnp.asarray(x_true), jnp.asarray(u_applied))[0])
        y = np.array(observation(jnp.asarray(x_true))) + rng.normal(0, np.sqrt(VAR))
        x_hat, P = ekf.update(jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(y))
        x_hat, P = np.array(x_hat), np.array(P)
        ef = x_hat[4:7] - x_true[4:7]
        Pf = P[4:7, 4:7]
        rec["lead"].append(x_true[0:3].copy())
        rec["ftrue"].append(x_true[4:7].copy())
        rec["fest"].append(x_hat[4:7].copy())
        rec["Pf"].append(Pf.copy())
        rec["err"].append(float(np.linalg.norm(ef)))
        rec["sig"].append(float(3 * np.sqrt(max(np.linalg.eigvalsh(Pf)[-1], 0.0))))
        rec["dist"].append(float(np.linalg.norm(x_true[4:7] - x_true[0:3])))
        t += DT
        if stream:
            rr.set_time("/time", duration=t)
            rr.log(
                "/sim/leader",
                rr.Points3D(x_true[0:3][None], colors=[0, 0, 0], radii=0.12),
            )
            rr.log(
                "/sim/follower_true",
                rr.Points3D(x_true[4:7][None], colors=[220, 40, 40], radii=0.12),
            )
            rr.log(
                "/sim/follower_est",
                rr.Points3D(x_hat[4:7][None], colors=[230, 160, 40], radii=0.1),
            )
            rr.log(
                "/sim/follower_est/cov3sigma",
                rr.LineStrips3D(cov_ellipsoid(x_hat[4:7], Pf), colors=[230, 160, 40]),
            )
            rr.log(
                "/sim/range",
                rr.LineStrips3D(
                    np.array([x_true[0:3], x_true[4:7]])[None], colors=[120, 120, 120]
                ),
            )
            rr.log("/graphs/error/oac", rr.Scalars(rec["err"][-1]))
            rr.log("/graphs/error/three_sigma", rr.Scalars(rec["sig"][-1]))
            if noac_err is not None:
                rr.log("/graphs/error/noac", rr.Scalars(float(noac_err[i])))
            rr.log("/graphs/observability/min_eig", rr.Scalars(rec["mineig"][-1]))
            rr.log("/graphs/solve/ms", rr.Scalars(rec["walls"][-1]))
            rr.log("/graphs/distance/dist", rr.Scalars(rec["dist"][-1]))
            for b, v in (("min", DIST_BOUNDS[0]), ("max", DIST_BOUNDS[1])):
                rr.log(f"/graphs/distance/{b}", rr.Scalars(v))
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
            rrb.Spatial3DView(
                origin="/sim", name="Flat-output leader-follower under OAC (range-only)"
            ),
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


def summarize_and_plot(oac, noac):
    tt = np.arange(STEPS) * DT
    print(
        "=== RT-OAC flat-output cooperative navigation (range-only, rung 3a) + carried EKF ==="
    )
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
    print(
        "=> OAC drives the unobservable (tangential) initial error down; no-OAC cannot."
    )

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        "RT-OAC flat-output [x,y,z,psi] cooperative navigation: range-only, softmin-eig "
        "STLOG + SLSQP@6 + carried EKF (OAC vs no-OAC)",
        fontsize=13,
    )

    ax = fig.add_subplot(2, 3, 1, projection="3d")
    ax.plot(*oac["lead"].T, color="black", lw=2, label="leader")
    ax.plot(*oac["ftrue"].T, color="tab:red", lw=1.6, label="follower true (OAC)")
    ax.plot(
        *oac["fest"].T, color="tab:orange", lw=1.1, ls="--", label="follower est (OAC)"
    )
    ax.plot(
        *noac["ftrue"].T,
        color="tab:gray",
        lw=1.4,
        ls=":",
        label="follower true (no-OAC)",
    )
    ax.set(title="Trajectories (3D)", xlabel="x", ylabel="y", zlabel="z")
    ax.legend(fontsize=7, loc="upper left")

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
    ax.set(
        title=f"Error: OAC {noac['final'] / max(oac['final'], 1e-9):.1f}x lower (final)",
        xticks=xb,
        ylabel="m",
    )
    ax.set_xticklabels(["no-OAC", "OAC"])
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 4)
    ax.plot(tt, oac["mineig"], color="tab:blue", lw=1.5)
    ax.set(
        title="Accumulated observability (min-eig)", xlabel="t [s]", ylabel="min-eig"
    )
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 3, 5)
    ax.plot(tt, oac["dist"], color="tab:purple", lw=1.5, label="inter-robot distance")
    ax.axhspan(DIST_BOUNDS[0], DIST_BOUNDS[1], color="green", alpha=0.12, label="band")
    ax.set(title="Inter-robot distance", xlabel="t [s]", ylabel="m")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 6)
    ax.plot(tt, oac["walls"], color="tab:green", lw=1.3)
    ax.axhline(100, color="0.6", ls="--", label="100 ms (10 Hz)")
    ax.set(title="Solve time", xlabel="t [s]", ylabel="ms")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = RESULTS / "example_flat_robot.png"
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


@hydra.main(version_base=None, config_path="conf", config_name="flat_robot")
def main(cfg: DictConfig):
    ctrl, cost = build_controller(cfg.soft)
    gram = jax.jit(
        lambda x, u: cost(x, jnp.asarray(u), STLOG_DT, return_gramians=True).gramians
    )

    rr.init("rt_oac_flat_robot", spawn=cfg.spawn)
    rrd = cfg.save or str(RESULTS / "example_flat_robot.rrd")
    if not cfg.spawn:
        rr.save(rrd)
    style_series()
    send_blueprint()

    noac = run(False, ctrl, gram, stream=False)
    oac = run(True, ctrl, gram, stream=True, noac_err=noac["err"])
    summarize_and_plot(oac, noac)

    if cfg.report:
        tag = "flat_robot" + ("_soft" if cfg.soft else "")
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
                "kind": "flat_robot",
                "objective": "softmin_eig",
                "soft": bool(cfg.soft),
                "steps": int(STEPS),
                "dt": float(DT),
                "band_lo": float(DIST_BOUNDS[0]),
                "band_hi": float(DIST_BOUNDS[1]),
                "oac_rmse_m": oac["rmse"],
                "oac_final_m": oac["final"],
                "noac_rmse_m": noac["rmse"],
                "noac_final_m": noac["final"],
                "plan_median_ms": float(np.median(oac["walls"][1:])),
            },
            arrays=arrays,
        )
        print(f"report data -> {cfg.report}/{tag}.{{npz,json}}")
    if not cfg.spawn:
        print(f"rerun recording written; open with:  rerun {rrd}")


if __name__ == "__main__":
    main()
