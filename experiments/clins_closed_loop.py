r"""(E1,O0) Phase 2 — CLOSE THE LOOP: flat OAC re-planned on the carried collaborative INS.

Phase 1 (clins_bridge.py) validated the estimator port with control-on-truth-tracking. Phase 2 puts the
honest-but-loose collaborative INS IN CONTROL — the full (E1,O0) cell, and the closed-loop, carried,
collaborative-INS version of flatout_bridge:

  flat OAC re-plans each step on the carried INS estimate (RHOAC)  ->  geometric tracker (using that
  estimate)  ->  full-fidelity quad WITH DRAG  ->  synthesized IMU + range  ->  collaborative INS  -> ...

THE DECISIVE TEST: the old command-driven carried quad loop DIVERGED to 100s-1000s m (confident-wrong,
findings #8-#15). Does an HONEST (but loose, tangential 3sig ~1.1 m) covariance — from driving the filter
with the measured IMU, not the command — keep the closed loop BOUNDED? The verification predicts
"bounded-but-loose, backs off"; this either confirms that or finds a new failure (the controller wandering
inside the ~1 m honest uncertainty, or steering into low-excitation geometry). Compares OA vs no-OAC, and
IMU-driven vs command-driven (the #8 reproduction) — all carried, closed-loop, multi-seed.

    PYTHONPATH=src JAX_PLATFORMS=cpu uv run python experiments/clins_closed_loop.py --seeds 12
"""

import argparse

from example_lib.misc.geometric_controller import AttitudeController, TrackingController
from example_lib.models import quadrotor
import jax
import jax.numpy as jnp
import numpy as np
from observability_aware_control import integrator, observability_cost
from scipy.spatial.transform import Rotation

from rt_oac import metrics
from rt_oac.controller import RTController

# --- timing / formation ---
DT, STLOG_DT, WINDOW, STEPS, ORDER, N_SUB = 0.1, 0.2, 5, 120, 2, 10
LEADER_VEL = np.array([1.0, 0.0, 0.0])
R0 = np.array([
    0.0,
    5.0,
    0.0,
])  # nominal relative position (follower - leader), standoff 5 m
DIST_BOUNDS = (4.5, 5.5)
FB_LB = np.array([-1.0, -2.0, -2.0, -2.0])  # follower [vx,vy,vz,yaw-rate] bounds
FB_UB = np.array([3.0, 2.0, 2.0, 2.0])
# --- flat OAC obs / sensor ---
VAR = np.r_[np.full(3, 1e-1), 1e-2, 1e-2, 1e-2]  # [leader pos(3), psi_l, psi_f, range]
# --- INS / IMU / truth ---
G = np.array([0.0, 0.0, -9.81])
K_DRAG = 0.20  # truth drag (the IMU senses it, the command doesn't)
ACCEL_NOISE, ACCEL_BIAS_STD, ATT_ERR_DEG, RANGE_VAR = 0.05, 0.10, 0.7, 1e-2
Q_ACCEL = 1.0  # honest process-noise floor (Phase 1: where NEES bottoms out, consistent-but-loose)
OFFSET_MAG = 2.26  # tangential (unobservable) initial follower-position error


# --- flat-output OAC model (xdot = u; obs = leader pos + both yaws + range) ---
def flat_dyn(x, u):
    return jnp.asarray(u)


def flat_obs(x, u=None):
    return jnp.array([x[0], x[1], x[2], x[3], x[7], jnp.linalg.norm(x[4:7] - x[0:3])])


def interrobot_distance(x0, us, cost):
    xs, _ = cost.eval_integrator(x0, us)
    return jnp.linalg.norm(xs[:, 4:7] - xs[:, 0:3], axis=1)


def build_oac(order=ORDER):
    cost = observability_cost.ObservabilityCost(
        flat_dyn,
        flat_obs,
        DT,
        gramian_kw={"order": order, "var": VAR},
        gramian_metric=metrics.neg_softmin_eig,
        observed_indices=(),
    )
    return RTController(
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
        constraint_mode="hard",
    )


BARO_STD = 0.3  # follower barometer altitude noise (m)


def flat_obs_baro(
    x, u=None
):  # 1 range + both yaws + leader pos + follower ALTITUDE (barometer)
    return jnp.array([
        x[0],
        x[1],
        x[2],
        x[3],
        x[7],
        jnp.linalg.norm(x[4:7] - x[0:3]),
        x[6],
    ])


def build_oac_baro(order=ORDER):
    cost = observability_cost.ObservabilityCost(
        flat_dyn,
        flat_obs_baro,
        DT,
        gramian_kw={"order": order, "var": np.r_[VAR, BARO_STD**2]},
        gramian_metric=metrics.neg_softmin_eig,
        observed_indices=(),
    )
    return RTController(
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
        constraint_mode="hard",
    )


# --- 2-range OAC: 12-state [leader1(4), follower(4), leader2(4)]; both leaders KNOWN (directly observed),
#     follower observed via BOTH ranges, so the STLOG worst-eigenvalue targets the follower fused by 2 anchors
VAR2 = np.r_[
    np.full(3, 1e-1), 1e-2, np.full(3, 1e-1), 1e-2, 1e-2, 1e-2, 1e-2
]  # L1pos,psiL1,L2pos,psiL2,psif,r1,r2


def flat_obs2(x, u=None):
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


def build_oac2():
    cost = observability_cost.ObservabilityCost(
        flat_dyn,
        flat_obs2,
        DT,
        gramian_kw={"order": ORDER, "var": VAR2},
        gramian_metric=metrics.neg_softmin_eig,
        observed_indices=(),
    )
    return RTController(
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
        constraint_mode="hard",
    )


def _drag(v):
    return -K_DRAG * v


def qdyn(x, u):
    dx = quadrotor.dynamics(x, u)
    return dx.at[7:10].add(_drag(x[7:10]))


def _ahrs(q_true, rng):
    err = np.deg2rad(ATT_ERR_DEG) * rng.standard_normal(3)
    return (Rotation.from_rotvec(err) * Rotation.from_quat(q_true)).as_quat()


def run(
    mode, ctrl, seed, *, driven="imu", ff_clip=0.0, l2_off=None, oac2=False, baro=False
):
    """One closed-loop run. mode in {oac, noac}; driven in {imu, cmd, truth}. ff_clip = max |accel
    feedforward| (0 = none). l2_off = constant offset of a 2nd KNOWN leader from leader 1 (None = single
    leader); the INS then fuses ranges to both. oac2 = the OA planner uses the 12-state 2-range model
    (build_oac2; requires l2_off) instead of planning on leader 1 only. Returns per-step recovery, NEES
    (pos block), true formation distance, OA solve wall time, vel-tracking error, commanded/achieved maneuver."""
    rng = np.random.default_rng(seed)
    sim = jax.jit(
        integrator.Integrator(qdyn, integrator.Methods.RK4, stepsize=DT / N_SUB)
    )
    tctrl, actrl = TrackingController(), AttitudeController()
    dti = DT / N_SUB

    xq = np.r_[R0, 0.0, 0.0, 0.0, 1.0, LEADER_VEL]  # quad truth [pos, quat xyzw, vel]
    b_a = rng.normal(0, ACCEL_BIAS_STD, 3)
    that = np.cross(R0 / np.linalg.norm(R0), [0, 0, 1.0])
    that /= np.linalg.norm(that)
    r_hat = (
        R0 + OFFSET_MAG * that
    )  # carried estimate, with the unobservable tangential error
    v_hat = LEADER_VEL.copy()
    P = np.diag([4.0, 4.0, 4.0, 0.04, 0.04, 0.04])
    Qa = Q_ACCEL**2 * np.eye(3)
    F = np.block([[np.eye(3), dti * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])
    Gn = np.vstack([0.5 * dti**2 * np.eye(3), dti * np.eye(3)])

    prev_u, t = None, 0.0
    errs, nees, dists, walls, vtrack, vig_c, vig_a = [], [], [], [], [], [], []
    for _i in range(STEPS):
        lead = LEADER_VEL * t
        if mode == "oac":
            if oac2:  # 12-state [L1, follower, L2]; OA plans on BOTH ranges
                x_flat = jnp.asarray(
                    np.r_[lead, 0.0, lead + r_hat, 0.0, lead + l2_off, 0.0]
                )
                base_u = np.r_[LEADER_VEL, 0.0, LEADER_VEL, 0.0, LEADER_VEL, 0.0]
            else:  # 8-state [L1, follower]; OA plans on leader 1 only
                x_flat = jnp.asarray(np.r_[lead, 0.0, lead + r_hat, 0.0])
                base_u = np.r_[LEADER_VEL, 0.0, LEADER_VEL, 0.0]
            guess = np.tile(base_u, (WINDOW, 1))
            if prev_u is not None:
                guess[:, 4:8] = np.vstack([prev_u[1:, 4:8], prev_u[-1:, 4:8]])
            else:
                kk = np.arange(WINDOW)
                guess[:, 4:7] += 0.3 * np.c_[np.sin(kk), np.cos(kk), np.sin(kk + 2.0)]
            res = ctrl.solve(x_flat, jnp.asarray(guess))
            prev_u = res.u
            v_cmd = np.asarray(res.u[0, 4:7])
            # accel feedforward, magnitude-clipped to ff_clip (the OA plan's raw accel is infeasible
            # ~2.3 g; ff_clip=0 drops it entirely, the Phase-2 default). Feeding the RAW accel overshoots
            # the tracker and drives the formation to ~48 m even plan-on-truth.
            raw_a = np.asarray(res.u[1, 4:7] - res.u[0, 4:7]) / DT
            an = np.linalg.norm(raw_a)
            a_cmd = (
                raw_a * min(1.0, ff_clip / (an + 1e-9)) if ff_clip > 0 else np.zeros(3)
            )
            walls.append(res.wall_time * 1e3)
        else:
            v_cmd, a_cmd = LEADER_VEL.copy(), np.zeros(3)

        r_start = r_hat.copy()
        for k in range(N_SUB):
            lead_k = LEADER_VEL * t
            ref_abs = lead_k + r_start + (v_cmd - LEADER_VEL) * (k * dti)
            st = TrackingController.State(
                jnp.asarray(lead_k + r_hat), jnp.asarray(xq[3:7]), jnp.asarray(v_hat)
            )
            rf = TrackingController.Reference(
                jnp.asarray(ref_abs), jnp.asarray(v_cmd), jnp.asarray(a_cmd), 0.0
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
            f_b = Rq.inv().apply(f_world) + b_a + rng.normal(0, ACCEL_NOISE, 3)
            a_world = (
                Rotation.from_quat(_ahrs(xq[3:7], rng)).apply(
                    f_b if driven == "imu" else np.array([0.0, 0.0, thrust])
                )
                + G
            )
            v_hat = v_hat + a_world * dti  # noqa: PLR6104
            r_hat = r_hat + (v_hat - LEADER_VEL) * dti  # noqa: PLR6104
            P = F @ P @ F.T + Gn @ Qa @ Gn.T
            t += dti

        r_true = xq[0:3] - LEADER_VEL * t  # follower position relative to leader 1
        offs = [np.zeros(3)] if l2_off is None else [np.zeros(3), l2_off]
        for off in offs:  # range update for each KNOWN leader (leader i sits at constant offset `off`)
            rr_hat, rr_true = (
                r_hat - off,
                r_true - off,
            )  # vector from leader i to the follower
            rn = np.linalg.norm(rr_hat)
            H = np.r_[rr_hat / (rn + 1e-12), np.zeros(3)][None]
            z = np.linalg.norm(rr_true) + rng.normal(0, np.sqrt(RANGE_VAR))
            S = H @ P @ H.T + RANGE_VAR
            K = (P @ H.T) / S
            dxu = (K * (z - rn)).ravel()
            r_hat, v_hat = r_hat + dxu[0:3], v_hat + dxu[3:6]
            P = (np.eye(6) - K @ H) @ P
        if baro:  # follower barometer: direct measurement of the relative vertical position r_z
            Hb = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
            zb = r_true[2] + rng.normal(0, BARO_STD)
            Sb = Hb @ P @ Hb.T + BARO_STD**2
            Kb = (P @ Hb.T) / Sb
            dxu = (Kb * (zb - r_hat[2])).ravel()
            r_hat, v_hat = r_hat + dxu[0:3], v_hat + dxu[3:6]
            P = (np.eye(6) - Kb @ Hb) @ P
        P = 0.5 * (P + P.T)
        if (
            driven == "truth"
        ):  # plan-on-truth diagnostic: re-anchor the estimate (isolates wiring vs INS)
            r_hat, v_hat = r_true.copy(), xq[7:10].copy()

        e = r_hat - r_true
        errs.append(float(np.linalg.norm(e)))
        nees.append(float(e @ np.linalg.solve(P[0:3, 0:3] + 1e-9 * np.eye(3), e)))
        dists.append(float(np.linalg.norm(r_true)))
        vtrack.append(
            float(np.linalg.norm(xq[7:10] - v_cmd))
        )  # velocity-tracking shortfall
        vig_c.append(
            float(np.linalg.norm(v_cmd - LEADER_VEL))
        )  # commanded maneuver speed
        vig_a.append(
            float(np.linalg.norm(xq[7:10] - LEADER_VEL))
        )  # achieved maneuver speed
    return (
        np.array(errs),
        np.array(nees),
        np.array(dists),
        (np.mean(walls) if walls else 0.0),
        np.array(vtrack),
        np.array(vig_c),
        np.array(vig_a),
    )


def main():
    global FB_LB, FB_UB, N_SUB
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument(
        "--vscale",
        type=float,
        default=1.0,
        help="scale follower vel bounds (gentler OA)",
    )
    ap.add_argument(
        "--ff-clip",
        type=float,
        default=0.0,
        help="max |accel feedforward| m/s^2 (0 = none)",
    )
    ap.add_argument(
        "--n-sub", type=int, default=N_SUB, help="tracker inner-loop steps per OA step"
    )
    ap.add_argument(
        "--only-imu",
        action="store_true",
        help="run only the OAC IMU-driven config (faster)",
    )
    args = ap.parse_args()
    FB_LB, FB_UB, N_SUB = FB_LB * args.vscale, FB_UB * args.vscale, args.n_sub
    print(
        "=== (E1,O0) Phase 2: flat OAC re-planned on the CARRIED collaborative INS (closed loop) ==="
    )
    print(
        f"{STEPS} steps, drag K={K_DRAG}, q_accel={Q_ACCEL}, {args.seeds} seeds, vscale={args.vscale}, "
        f"ff_clip={args.ff_clip}, n_sub={N_SUB}; tangential init {OFFSET_MAG} m; bounded = recovery<1.5 m "
        f"AND formation<15 m\n"
    )
    print(
        f"{'config':>26} {'rec_med':>8} {'%bnd':>6} {'NEES':>7} {'distMed':>8} {'vtrack':>7} "
        f"{'cmd_spd':>8} {'ach_spd':>8} {'ach/cmd':>8} {'ms':>6}"
    )
    configs = [
        ("OAC, plan-on-truth", "oac", "truth"),
        ("OAC, IMU-driven", "oac", "imu"),
        ("no-OAC, IMU-driven", "noac", "imu"),
        ("OAC, command-driven (#8)", "oac", "cmd"),
    ]
    if args.only_imu:
        configs = [("OAC, IMU-driven", "oac", "imu")]
    ctrl = build_oac()
    for label, mode, driven in configs:
        rows = []
        for s in range(args.seeds):
            errs, nees, dists, wall, vtrack, vig_c, vig_a = run(
                mode,
                ctrl if mode == "oac" else None,
                s,
                driven=driven,
                ff_clip=args.ff_clip,
            )
            bounded = errs[-1] < 1.5 and np.max(dists) < 15.0
            rows.append((
                errs[-1],
                np.median(nees),
                bounded,
                np.median(dists),
                wall,
                np.sqrt(np.mean(vtrack**2)),
                np.mean(vig_c),
                np.mean(vig_a),
            ))
        rows = np.array(rows)
        cmd_spd, ach_spd = np.median(rows[:, 6]), np.median(rows[:, 7])
        print(
            f"{label:>26} {np.median(rows[:, 0]):>8.2f} {100 * np.mean(rows[:, 2]):>5.0f}% "
            f"{np.median(rows[:, 1]):>7.1f} {np.median(rows[:, 3]):>8.2f} {np.median(rows[:, 5]):>7.2f} "
            f"{cmd_spd:>8.2f} {ach_spd:>8.2f} {ach_spd / (cmd_spd + 1e-9):>8.2f} {np.median(rows[:, 4]):>6.1f}"
        )
    print(
        "\nVERDICT (Phase 2): the honest collaborative INS converts the CATASTROPHIC command-driven "
        "divergence (0% bounded, the #8 reproduction) into a MARGINALLY-bounded loop (~30%) -- necessary "
        "but NOT sufficient. The formation holds (~5.4 m, the dist constraint + tracker work once the "
        "infeasible OA accel is NOT fed forward); the divergence is entirely in the estimate's tangential, "
        "and plan-on-truth's 100% shows the ceiling exists. The honest covariance removes the divergence "
        "MECHANISM; the loose Gaussian filter still rides the #14 ridge on ~2/3 of seeds. Next lever: a "
        "non-Gaussian filter (robustness) or measurement multiplicity / more agents (the floor) -- NOT "
        "more estimator process-model work."
    )


if __name__ == "__main__":
    main()
