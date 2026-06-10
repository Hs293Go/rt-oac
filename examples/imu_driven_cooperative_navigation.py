r"""Headline example: real-time (E1, O0) cooperative navigation -- IMU-driven INS, CLOSED loop, with rerun.

The deployable corner of the range-only OAC program. Above the rung-3a flat-output headline
(``flat_robot_cooperative_navigation.py``, which plans on a carried *simple* EKF over a flat point-pair),
this example closes the loop on the full deployment stack:

  flat-output O0 OA re-plans each step on a CARRIED collaborative INS (E1)  ->  geometric tracker realizes
  the flat velocity on a FULL quadrotor (truth, with drag)  ->  synthesized IMU (specific force + body
  rate) + inter-robot ranges drive the INS  ->  ... (receding horizon).

Why this is the deployable corner, and why each piece earns its place (rt-oac findings #8-#15, doc
``quad_climb_guardrails.md`` sections 6.1-6.6):

* **E1 (IMU-driven estimator).** The naive loop feeds the *control command* to the estimator; the carried
  filter then trusts a model the drone is not actually flying (drag, tracking lag) and turns
  confident-wrong -- the documented divergence to 100s-1000s of m. Driving the INS with the *measured* IMU
  (specific force + gyro) instead removes that mechanism: the filter sees what the vehicle really did.
* **O0 (flat-output objective).** By differential flatness any smooth ``sigma=[x,y,z,psi]`` is
  quad-realizable, so the OA plans in the smallest honest planning space (``xdot=u``); the full-quad O1
  objective buys nothing here once the estimator measures velocity+attitude (section 6.12) and only adds
  fragility. The metric is the **soft-min-eigenvalue STLOG** (E-optimality), bounded standoff, SLSQP@6.
* **TWO known leaders + 2-range OA.** One scalar range leaves the follower's tangential position on a weak
  ridge (~25-38% bounded even with the honest INS). A 2nd known anchor removes the ridge by geometry, and
  the 12-state 2-range planner exploits both -- lifting the closed loop to **~95-100% bounded**, recovery
  ~0.25 m, NEES ~5 (consistent), at a few ms/solve. This is the validated, carried, no-truth-in-the-loop
  configuration.

Compares OA against a no-OAC follower (cruises straight alongside) on closed-loop INS accuracy, from a
large initial follower-position error in the *unobservable* (tangential) direction -- OA drives it down
where no-OAC cannot.

Visualization is **rerun**: a live 3D scene (both leaders, true follower, INS estimate + its shrinking
3-sigma ellipsoid, the two range links) plus live time-series (OA-vs-no-OAC error & 3-sigma, NEES,
accumulated observability, solve time, inter-robot distance). A multi-panel matplotlib figure follows.

Live viewer:  uv run python examples/imu_driven_cooperative_navigation.py spawn=true
Headless:     uv run python examples/imu_driven_cooperative_navigation.py
"""

import pathlib

from example_lib.misc.geometric_controller import AttitudeController, TrackingController
from example_lib.models import quadrotor
import hydra
import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
from observability_aware_control import integrator, observability_cost
from omegaconf import DictConfig
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial.transform import Rotation
import tqdm

import rt_oac  # noqa: F401
from rt_oac import metrics, report
from rt_oac.controller import RTController

mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --- timing / formation ---
DT, STLOG_DT, WINDOW, STEPS, ORDER, N_SUB = 0.1, 0.2, 5, 120, 2, 10
LEADER_VEL = np.array([1.0, 0.0, 0.0])  # leader 1 cruises +x
R0 = np.array([
    0.0,
    5.0,
    0.0,
])  # nominal relative position (follower - leader 1), 5 m standoff
D12 = np.array([
    0.0,
    10.0,
    0.0,
])  # 2nd KNOWN leader, lateral offset from leader 1 (the ~95% placement)
DIST_BOUNDS = (4.5, 5.5)
FB_LB = np.array([-1.0, -2.0, -2.0, -2.0])  # follower [vx, vy, vz, yaw-rate] bounds
FB_UB = np.array([3.0, 2.0, 2.0, 2.0])
OFFSET_MAG = 2.26  # tangential (unobservable) initial follower-position error
# --- INS / IMU / truth ---
G = np.array([0.0, 0.0, -9.81])
K_DRAG = 0.20  # truth drag (the IMU senses it; the flat command does not)
ACCEL_NOISE, ACCEL_BIAS_STD, ATT_ERR_DEG, RANGE_VAR = 0.05, 0.10, 0.7, 1e-2
Q_ACCEL = 1.0  # honest INS process-noise floor (consistent-but-loose, section 6.1)
# --- 2-range flat OAC over [leader1(4), follower(4), leader2(4)]; both leaders KNOWN ---
VAR2 = np.r_[np.full(3, 1e-1), 1e-2, np.full(3, 1e-1), 1e-2, 1e-2, 1e-2, 1e-2]
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"


def flat_dyn(x, u):
    return jnp.asarray(u)  # flat-output kinematics: xdot = u


def flat_obs2(x, u=None):
    """12-state observation: leader poses (known), both yaws, and the two inter-robot ranges."""
    foll = x[4:7]
    return jnp.array([
        x[0],
        x[1],
        x[2],
        x[3],
        x[8],
        x[9],
        x[10],
        x[11],
        x[7],
        jnp.linalg.norm(foll - x[0:3]),
        jnp.linalg.norm(foll - x[8:11]),
    ])


def interrobot_distance(x0, us, cost):
    xs, _ = cost.eval_integrator(x0, us)
    return jnp.linalg.norm(xs[:, 4:7] - xs[:, 0:3], axis=1)


def build_controller():
    cost = observability_cost.ObservabilityCost(
        flat_dyn,
        flat_obs2,
        DT,
        gramian_kw={"order": ORDER, "var": VAR2},
        gramian_metric=metrics.neg_softmin_eig,
        observed_indices=(),
    )
    ctrl = RTController(
        cost,
        stlog_dt=STLOG_DT,
        lb=FB_LB,
        ub=FB_UB,
        n_inputs=12,
        follower_indices=(4, 5, 6, 7),
        method="SLSQP",
        maxiter=6,
        constraint=interrobot_distance,
        constraint_bounds=DIST_BOUNDS,
    )
    return ctrl, cost


def _drag(v):
    return -K_DRAG * v


def qdyn(
    x, u
):  # full quad truth: rigid-body dynamics + linear drag (sensed by the IMU, not the command)
    return quadrotor.dynamics(x, u).at[7:10].add(_drag(x[7:10]))


def _ahrs(q_true, rng):  # AHRS attitude estimate = truth + small rotation noise
    err = np.deg2rad(ATT_ERR_DEG) * rng.standard_normal(3)
    return (Rotation.from_rotvec(err) * Rotation.from_quat(q_true)).as_quat()


def cov_ellipsoid(center3, P3, nsig=3.0, n=24):
    """3-sigma covariance ellipsoid as 3 orthogonal principal-plane rings (each (n,3))."""
    vals, vecs = np.linalg.eigh(0.5 * (P3 + P3.T))
    axes = vecs * (nsig * np.sqrt(np.clip(vals, 0.0, None)))
    t = np.linspace(0, 2 * np.pi, n)
    rings = []
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        rings.append(
            (axes[:, i][None].T * np.cos(t) + axes[:, j][None].T * np.sin(t)).T
            + center3
        )
    return rings


def _acc_min_eig(gram, x, u):
    g = np.asarray(gram(jnp.asarray(x), u))
    acc = 0.5 * (g + np.swapaxes(g, -1, -2)).sum(0)
    return float(np.linalg.eigvalsh(acc)[0])


def run(use_oac, ctrl, gram, seed=0, *, stream=False, noac_err=None):
    """One closed-loop (E1, O0) run on a CARRIED collaborative INS. If ``stream``, log live to rerun."""
    rng = np.random.default_rng(seed)
    sim = jax.jit(
        integrator.Integrator(qdyn, integrator.Methods.RK4, stepsize=DT / N_SUB)
    )
    tctrl, actrl = TrackingController(), AttitudeController()
    dti = DT / N_SUB

    xq = np.r_[
        R0, 0.0, 0.0, 0.0, 1.0, LEADER_VEL
    ]  # quad truth [pos(3), quat xyzw, vel(3)]
    b_a = rng.normal(0, ACCEL_BIAS_STD, 3)
    that = np.cross(R0 / np.linalg.norm(R0), [0.0, 0.0, 1.0])
    that /= np.linalg.norm(that)
    r_hat = (
        R0 + OFFSET_MAG * that
    )  # carried estimate, with the unobservable tangential error
    v_hat = LEADER_VEL.copy()
    P = np.diag([4.0, 4.0, 4.0, 0.04, 0.04, 0.04])
    Qa = Q_ACCEL**2 * np.eye(3)
    F = np.block([[np.eye(3), dti * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])
    Gn = np.vstack([0.5 * dti**2 * np.eye(3), dti * np.eye(3)])

    rec = {
        k: []
        for k in (
            "lead",
            "lead2",
            "ftrue",
            "fest",
            "Pf",
            "err",
            "sig",
            "nees",
            "mineig",
            "dist",
            "walls",
        )
    }
    prev_u, t = None, 0.0
    for i in tqdm.trange(STEPS, disable=not stream):
        lead = LEADER_VEL * t
        if use_oac:
            x_flat = jnp.asarray(
                np.r_[lead, 0.0, lead + r_hat, 0.0, lead + D12, 0.0]
            )  # 12-state
            base_u = np.r_[LEADER_VEL, 0.0, LEADER_VEL, 0.0, LEADER_VEL, 0.0]
            guess = np.tile(base_u, (WINDOW, 1))
            if prev_u is not None:
                guess[:, 4:8] = np.vstack([prev_u[1:, 4:8], prev_u[-1:, 4:8]])
            else:  # symmetry-break the straight-line critical point of the observability objective
                kk = np.arange(WINDOW)
                guess[:, 4:7] += 0.3 * np.c_[np.sin(kk), np.cos(kk), np.sin(kk + 2.0)]
            res = ctrl.solve(x_flat, jnp.asarray(guess))  # acts on the carried ESTIMATE
            prev_u = res.u
            v_cmd = np.asarray(res.u[0, 4:7])
            rec["walls"].append(res.wall_time * 1e3)
            rec["mineig"].append(_acc_min_eig(gram, x_flat, res.u))
        else:
            v_cmd = (
                LEADER_VEL.copy()
            )  # cruise straight alongside (no observability seeking)

        r_start = r_hat.copy()
        for k in range(
            N_SUB
        ):  # geometric tracker + quad truth + IMU-driven INS, inner loop
            lead_k = LEADER_VEL * t
            ref_abs = lead_k + r_start + (v_cmd - LEADER_VEL) * (k * dti)
            st = TrackingController.State(
                jnp.asarray(lead_k + r_hat), jnp.asarray(xq[3:7]), jnp.asarray(v_hat)
            )
            rf = TrackingController.Reference(
                jnp.asarray(ref_abs), jnp.asarray(v_cmd), jnp.zeros(3), 0.0
            )
            out, _ = tctrl.run(st, rf)
            ar, _ = actrl.run(
                AttitudeController.State(jnp.asarray(_ahrs(xq[3:7], rng))),
                AttitudeController.Reference(out.orientation),
            )
            thrust = float(out.thrust)
            xq = np.array(
                sim(
                    jnp.asarray(xq),
                    jnp.asarray(np.r_[thrust, np.asarray(ar.body_rate)]),
                )[0]
            )
            xq[3:7] /= np.linalg.norm(xq[3:7])
            Rq = Rotation.from_quat(xq[3:7])
            f_world = Rq.apply([0.0, 0.0, thrust]) + _drag(
                xq[7:10]
            )  # true specific force (incl. drag)
            f_b = (
                Rq.inv().apply(f_world) + b_a + rng.normal(0, ACCEL_NOISE, 3)
            )  # measured IMU specific force
            a_world = (
                Rotation.from_quat(_ahrs(xq[3:7], rng)).apply(f_b) + G
            )  # de-rotated by the AHRS
            v_hat = v_hat + a_world * dti  # noqa: PLR6104  (INS strapdown integration)
            r_hat = r_hat + (v_hat - LEADER_VEL) * dti  # noqa: PLR6104
            P = F @ P @ F.T + Gn @ Qa @ Gn.T
            t += dti

        r_true = xq[0:3] - LEADER_VEL * t  # follower position relative to leader 1
        for off in (
            np.zeros(3),
            D12,
        ):  # range update for each KNOWN leader (at constant offset `off`)
            rr_hat, rr_true = r_hat - off, r_true - off
            rn = np.linalg.norm(rr_hat)
            H = np.r_[rr_hat / (rn + 1e-12), np.zeros(3)][None]
            z = np.linalg.norm(rr_true) + rng.normal(0, np.sqrt(RANGE_VAR))
            S = H @ P @ H.T + RANGE_VAR
            K = (P @ H.T) / S
            dxu = (K * (z - rn)).ravel()
            r_hat, v_hat = r_hat + dxu[0:3], v_hat + dxu[3:6]
            P = (np.eye(6) - K @ H) @ P
        P = 0.5 * (P + P.T)

        e = r_hat - r_true
        Pf = P[0:3, 0:3]
        rec["lead"].append((LEADER_VEL * t).copy())
        rec["lead2"].append((LEADER_VEL * t + D12).copy())
        rec["ftrue"].append(xq[0:3].copy())
        rec["fest"].append((LEADER_VEL * t + r_hat).copy())
        rec["Pf"].append(Pf.copy())
        rec["err"].append(float(np.linalg.norm(e)))
        rec["sig"].append(float(3 * np.sqrt(max(np.linalg.eigvalsh(Pf)[-1], 0.0))))
        rec["nees"].append(float(e @ np.linalg.solve(Pf + 1e-9 * np.eye(3), e)))
        rec["dist"].append(float(np.linalg.norm(r_true)))
        if stream:
            _stream(rec, i, noac_err)
    out = {k: np.array(v) for k, v in rec.items() if v}
    out["rmse"] = float(np.sqrt((out["err"] ** 2).mean()))
    out["final"] = float(out["err"][-1])
    out["bounded"] = bool(out["final"] < 1.5 and out["dist"].max() < 15.0)
    return out


def _stream(rec, i, noac_err):
    lead, lead2, ft, fe, Pf = (
        rec["lead"][-1],
        rec["lead2"][-1],
        rec["ftrue"][-1],
        rec["fest"][-1],
        rec["Pf"][-1],
    )
    rr.set_time("/time", duration=(i + 1) * DT)
    rr.log("/sim/leader1", rr.Points3D(lead[None], colors=[0, 0, 0], radii=0.13))
    rr.log("/sim/leader2", rr.Points3D(lead2[None], colors=[60, 60, 60], radii=0.13))
    rr.log(
        "/sim/follower_true", rr.Points3D(ft[None], colors=[220, 40, 40], radii=0.12)
    )
    rr.log("/sim/follower_est", rr.Points3D(fe[None], colors=[230, 160, 40], radii=0.1))
    rr.log(
        "/sim/follower_est/cov3sigma",
        rr.LineStrips3D(cov_ellipsoid(fe, Pf), colors=[230, 160, 40]),
    )
    rr.log(
        "/sim/range1",
        rr.LineStrips3D(np.array([lead, ft])[None], colors=[120, 120, 120]),
    )
    rr.log(
        "/sim/range2",
        rr.LineStrips3D(np.array([lead2, ft])[None], colors=[120, 120, 120]),
    )
    rr.log("/graphs/error/oac", rr.Scalars(rec["err"][-1]))
    rr.log("/graphs/error/three_sigma", rr.Scalars(rec["sig"][-1]))
    if noac_err is not None:
        rr.log("/graphs/error/noac", rr.Scalars(float(noac_err[i])))
    rr.log("/graphs/nees/nees", rr.Scalars(rec["nees"][-1]))
    rr.log("/graphs/nees/expected", rr.Scalars(3.0))
    if rec["mineig"]:
        rr.log("/graphs/observability/min_eig", rr.Scalars(rec["mineig"][-1]))
    if rec["walls"]:
        rr.log("/graphs/solve/ms", rr.Scalars(rec["walls"][-1]))
    rr.log("/graphs/distance/dist", rr.Scalars(rec["dist"][-1]))
    for b, v in (("min", DIST_BOUNDS[0]), ("max", DIST_BOUNDS[1])):
        rr.log(f"/graphs/distance/{b}", rr.Scalars(v))


def style_series():
    spec = {
        "/graphs/error/oac": ("OAC error [m]", [230, 40, 40]),
        "/graphs/error/noac": ("no-OAC error [m]", [150, 150, 150]),
        "/graphs/error/three_sigma": ("OAC 3-sigma [m]", [230, 160, 40]),
        "/graphs/nees/nees": ("position NEES", [40, 120, 230]),
        "/graphs/nees/expected": ("expected (3 dof)", [150, 150, 150]),
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
                origin="/sim",
                name="(E1,O0) IMU-driven cooperative navigation (2 leaders)",
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(
                    origin="/graphs/error", name="Follower-position error & 3-sigma"
                ),
                rrb.TimeSeriesView(
                    origin="/graphs/nees", name="Position NEES (consistency)"
                ),
                rrb.TimeSeriesView(
                    origin="/graphs/observability", name="Accumulated observability"
                ),
                rrb.TimeSeriesView(origin="/graphs/solve", name="Solve time [ms]"),
                rrb.TimeSeriesView(
                    origin="/graphs/distance", name="Inter-robot distance [m]"
                ),
                row_shares=[1, 1, 1, 1, 1],
            ),
            column_shares=[1.4, 1.0],
        )
    )


def summarize_and_plot(oac, noac):
    tt = np.arange(STEPS) * DT
    print(
        "=== RT-OAC (E1,O0) IMU-driven cooperative navigation (range-only, 2 leaders, CLOSED loop) ==="
    )
    print("follower-position estimation error   |   no-OAC   |    OAC")
    print(
        f"  RMSE  [m]                          | {noac['rmse']:8.3f}   | {oac['rmse']:8.3f}"
    )
    print(
        f"  final [m]                          | {noac['final']:8.3f}   | {oac['final']:8.3f}"
    )
    print(
        f"  median position NEES (~3 = OK)     | {np.median(noac['nees']):8.1f}   | {np.median(oac['nees']):8.1f}"
    )
    print(
        f"  median solve time [ms]             |     --     | {np.median(oac['walls']):8.1f}"
    )
    print(
        "=> OAC sharpens recovery of the unobservable (tangential) error on a carried IMU-driven INS, "
        "real-time. This is ONE illustrative rollout; over >=20 seeds the deployable headline is "
        "~95-100% bounded (OAC) vs ~15% (no-OAC) -- the OA maneuver, not luck, closes the loop."
    )

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        "RT-OAC (E1,O0): IMU-driven collaborative INS + 2-range softmin-eig STLOG + geometric tracker, "
        "closed loop (OAC vs no-OAC)",
        fontsize=12,
    )
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    ax.plot(*oac["lead"].T, color="black", lw=2, label="leader 1")
    ax.plot(*oac["lead2"].T, color="0.4", lw=1.4, ls="-.", label="leader 2")
    ax.plot(*oac["ftrue"].T, color="tab:red", lw=1.6, label="follower true (OAC)")
    ax.plot(
        *oac["fest"].T, color="tab:orange", lw=1.0, ls="--", label="follower est (OAC)"
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
    ax.plot(tt, oac["nees"], color="tab:blue", lw=1.3, label="OAC position NEES")
    ax.axhline(3.0, color="0.6", ls="--", label="expected (3 dof)")
    ax.set(title="Estimator consistency (NEES)", xlabel="t [s]", ylabel="NEES")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 5)
    ax.plot(tt, oac["dist"], color="tab:purple", lw=1.5, label="inter-robot distance")
    ax.axhspan(DIST_BOUNDS[0], DIST_BOUNDS[1], color="green", alpha=0.12, label="band")
    ax.set(title="Inter-robot distance (leader 1)", xlabel="t [s]", ylabel="m")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 6)
    ax.plot(tt, oac["walls"], color="tab:green", lw=1.3)
    ax.axhline(100, color="0.6", ls="--", label="100 ms (10 Hz)")
    ax.set(title="Solve time", xlabel="t [s]", ylabel="ms")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = RESULTS / "example_imu_driven.png"
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


@hydra.main(version_base=None, config_path="conf", config_name="imu_driven")
def main(cfg: DictConfig):
    ctrl, cost = build_controller()
    gram = jax.jit(
        lambda x, u: cost(x, jnp.asarray(u), STLOG_DT, return_gramians=True).gramians
    )

    rr.init("rt_oac_imu_driven", spawn=cfg.spawn)
    rrd = cfg.save or str(RESULTS / "example_imu_driven.rrd")
    if not cfg.spawn:
        RESULTS.mkdir(exist_ok=True)
        rr.save(rrd)
    style_series()
    send_blueprint()

    noac = run(False, ctrl, gram, seed=cfg.seed, stream=False)
    oac = run(True, ctrl, gram, seed=cfg.seed, stream=True, noac_err=noac["err"])
    summarize_and_plot(oac, noac)

    if cfg.report:
        arrays = {
            f"oac_{k}": np.asarray(v)
            for k, v in oac.items()
            if k not in {"rmse", "final", "bounded"}
        }
        arrays.update({
            f"noac_{k}": np.asarray(v)
            for k, v in noac.items()
            if k not in {"rmse", "final", "bounded"}
        })
        arrays["t"] = np.arange(STEPS) * DT
        report.dump(
            cfg.report,
            "imu_driven",
            summary={
                "kind": "imu_driven_e1_o0",
                "objective": "softmin_eig_2range",
                "estimator": "imu_driven_collaborative_ins",
                "leaders": 2,
                "steps": int(STEPS),
                "dt": float(DT),
                "oac_rmse_m": oac["rmse"],
                "oac_final_m": oac["final"],
                "oac_median_nees": float(np.median(oac["nees"])),
                "noac_final_m": noac["final"],
                "plan_median_ms": float(np.median(oac["walls"][1:])),
            },
            arrays=arrays,
        )
        print(f"report data -> {cfg.report}/imu_driven.{{npz,json}}")
    if not cfg.spawn:
        print(f"rerun recording written; open with:  rerun {rrd}")


if __name__ == "__main__":
    main()
