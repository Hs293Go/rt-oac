r"""(E1,O1) corner with TWO LEADERS -- step 1: the 2-range full-quad OA in plan-on-truth.

The corner: the OA plans the follower's thrust + body-rates DIRECTLY by maximizing the full relative-pose
STLOG (`inter_quadrotor_pose`, order 5, manifold) over TWO ranges, on a carried IMU-driven relative-pose
ESEKF. Step 1 (this file) validates the OA machinery in plan-on-truth (perfect feedback, relative-pose
rollout): does the order-5, 2-range O1 OA solve, plan a sane observability maneuver, and HOLD the formation
(dist constraint), vs a no-OAC hover-hold? Step 2 will add the carried ESEKF (the closed corner).

State x = [r_lf(3), q_fl(4), v_lf(3)] (relative pose to leader 1). Input u = [f_l, w_l(3), f_f, w_f(3)]
(leader 1 + follower thrust/body-rates); leader 1 holds hover, the OA controls the follower (indices 4:8).
2nd leader sits at constant relative offset D12 (both leaders cruise level), so range2 = |r_lf - D12|.
Observed components (the OA's observability TARGET) = position + velocity [0,1,2,7,8,9] (attitude measured).

    PYTHONPATH=src JAX_PLATFORMS=cpu uv run python experiments/o1_corner.py
"""

import argparse

from example_lib import math as elmath
from example_lib.models import inter_quadrotor_pose as mdl, quadrotor
import jax
import jax.numpy as jnp
import numpy as np
from observability_aware_control import integrator, observability_cost

from rt_oac import metrics
from rt_oac.controller import RTController
from rt_oac.error_state_ekf import ErrorStateEKF

ORDER, STLOG_DT, WINDOW, DT, STEPS = 5, 0.2, 20, 0.05, 80
RANGE_VAR, ATT_VAR, GRAV = 0.01, 0.01, 9.81
LB = np.array([0.0, -4.0, -4.0, -2.0])  # follower [thrust, wx, wy, wz] bounds
UB = np.array([15.0, 4.0, 4.0, 6.0])
DIST2 = (1.0**2, 3.0**2)  # inter-vehicle distance band (squared), [1, 3] m
OBS_IDX = (
    0,
    1,
    2,
    7,
    8,
    9,
)  # position + velocity -- the structurally-weak observability target
U_HOVER = np.array([GRAV, 0.0, 0.0, 0.0])  # per-agent hover (mass 1)
D12_ABS = np.array([
    2.0,
    0.0,
    1.5,
])  # 2nd leader ABSOLUTE offset from leader 1 (non-collinear, 3D)
VAR2 = np.r_[RANGE_VAR, RANGE_VAR, np.full(4, ATT_VAR)]  # [range1^2, range2^2, q_fl(4)]
# initial relative pose: follower ~2 m from leader 1, level, at rest
X0 = np.array([0.0, 2.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
# --- step-2 (closed loop) constants ---
K_DRAG = 0.20  # truth drag (IMU partly senses; lateral part unmodeled by the rel model -> proc_cov)
ACCEL_NOISE, ACCEL_BIAS_STD, GYRO_NOISE = 0.05, 0.10, np.deg2rad(0.05)
VEL_VAR = 0.1**2  # IMU-derived relative-velocity measurement variance (lean estimator)
PROC_FLOOR = np.diag(
    np.r_[np.full(3, 0.02**2), np.full(3, 0.01**2), np.full(3, 0.15**2)]
)  # pos/att/vel


def obs2(
    x, u=None
):  # 2-range + relative-attitude obs; 2nd-leader offset rotated by relative attitude
    r_lf, q_fl = x[0:3], x[3:7]
    r2 = r_lf + elmath.quaternion_rotate_point(q_fl, jnp.asarray(D12_ABS))
    return jnp.concatenate([jnp.array([jnp.dot(r_lf, r_lf), jnp.dot(r2, r2)]), q_fl])


def obs_lean(
    x, u=None
):  # LEAN estimator obs: 2 ranges + attitude (AHRS) + velocity (IMU-provided)
    return jnp.concatenate([obs2(x), x[7:10]])  # [range1^2, range2^2, q_fl, v_lf]


class TwoRangeESEKF(ErrorStateEKF):
    """ESEKF with a 2-range + relative-attitude residual: [range1^2, range2^2, q_fl] -> 5-d residual."""

    def _residual(self, x, y):
        hx = self._observation(x)
        rng_resid = y[0:2] - hx[0:2]
        att_resid = elmath.quaternion_log(
            elmath.quaternion_product(elmath.quaternion_inverse(hx[2:6]), y[2:6])
        )
        return jnp.concatenate([rng_resid, att_resid])


class LeanESEKF(TwoRangeESEKF):
    """Lean estimator: velocity + attitude are MEASURED (IMU/AHRS), only the tangential position is
    range-aided. Residual [range1, range2, att(3), vel(3)] (8-d). Avoids the full ESEKF's range-only
    velocity estimation (the #14 ridge) -- the 'IMU-driven estimator is acceptable' architecture."""

    def _residual(self, x, y):
        hx = self._observation(x)
        rng_resid = y[0:2] - hx[0:2]
        att_resid = elmath.quaternion_log(
            elmath.quaternion_product(elmath.quaternion_inverse(hx[2:6]), y[2:6])
        )
        vel_resid = y[6:9] - hx[6:9]
        return jnp.concatenate([rng_resid, att_resid, vel_resid])


def _quad_drag(
    x, u
):  # absolute follower truth: quad dynamics + linear drag (unmodeled by the rel model)
    return quadrotor.dynamics(x, u).at[7:10].add(-K_DRAG * x[7:10])


def build_o1_oac(
    two_range=True, order=ORDER, maxiter=6, window=WINDOW, rate_scale=1.0
):  # maxiter 6 = frontier early-stop; rate_scale < 1 = gentler body-rate command (smoother for the ESEKF)
    lb = LB * np.r_[1.0, rate_scale, rate_scale, rate_scale]
    ub = UB * np.r_[1.0, rate_scale, rate_scale, rate_scale]
    obs = obs2 if two_range else mdl.observation
    var = VAR2 if two_range else np.r_[RANGE_VAR, np.full(4, ATT_VAR)]
    cost = observability_cost.ObservabilityCost(
        mdl.dynamics,
        obs,
        DT,
        gramian_kw={"order": order, "var": var, "manifold": mdl.MANIFOLD},
        gramian_metric=metrics.neg_softmin_eig,
        observed_indices=OBS_IDX,
    )
    ctrl = RTController(
        cost,
        stlog_dt=STLOG_DT,
        lb=lb,
        ub=ub,
        n_inputs=8,
        follower_indices=(4, 5, 6, 7),
        method="SLSQP",
        maxiter=maxiter,
        constraint=mdl.interrobot_distance_squared,
        constraint_bounds=DIST2,
    )
    return ctrl, cost


def run(ctrl, *, oac=True):
    sim = jax.jit(
        integrator.Integrator(mdl.dynamics, integrator.Methods.RK4, stepsize=DT)
    )
    x = X0.copy()
    prev_u = None
    dists, walls, _mineigs = [], [], []
    for _i in range(STEPS):
        if oac:
            guess = np.tile(np.r_[U_HOVER, U_HOVER], (WINDOW, 1))
            if prev_u is not None:
                guess[:, 4:8] = np.vstack([prev_u[1:, 4:8], prev_u[-1:, 4:8]])
            res = ctrl.solve(jnp.asarray(x), jnp.asarray(guess))
            prev_u = res.u
            u_foll = np.asarray(res.u[0, 4:8])
            walls.append(res.wall_time * 1e3)
        else:
            u_foll = U_HOVER.copy()
        u = np.r_[U_HOVER, u_foll]
        x = np.array(sim(jnp.asarray(x), jnp.asarray(u))[0])
        x[3:7] /= np.linalg.norm(x[3:7])
        dists.append(float(np.linalg.norm(x[0:3])))
    return np.array(dists), (np.mean(walls) if walls else 0.0)


def _att_noise(
    q, rng
):  # perturb a quaternion by a small rotation (std sqrt(ATT_VAR) rad)
    dv = np.sqrt(ATT_VAR) * rng.standard_normal(3)
    return np.array(elmath.quaternion_product(jnp.asarray(q), jnp.r_[dv / 2.0, 1.0]))


def closed_loop(
    ctrl, seed, *, driven="imu", offset_mag=2.0, proc_scale=1.0, lean=False
):
    """The (E1,O1) corner: O1 OA re-planned on a carried IMU-driven relative-pose ESEKF. lean=True uses the
    LEAN estimator (velocity+attitude MEASURED via IMU/AHRS, only the tangential position range-aided -- the
    'IMU-driven estimator is acceptable' fix); lean=False is the full range-only ESEKF (ridge-fragile).
    Leaders hover; follower truth is a full quad with drag; the ESEKF predicts with the measured follower
    IMU (E1). Returns per-step recovery |p_f_hat - p_f_true|, position-NEES, formation distance."""
    rng = np.random.default_rng(seed)
    qsim = jax.jit(
        integrator.Integrator(_quad_drag, integrator.Methods.RK4, stepsize=DT)
    )
    in_cov = np.diag(
        np.r_[1e-4, np.full(3, 1e-4), ACCEL_NOISE**2, np.full(3, GYRO_NOISE**2)]
    )
    if lean:
        ekf = LeanESEKF(
            mdl.dynamics,
            obs_lean,
            mdl.MANIFOLD,
            in_cov=in_cov,
            obs_cov=np.diag(
                np.r_[RANGE_VAR, RANGE_VAR, np.full(3, ATT_VAR), np.full(3, VEL_VAR)]
            ),
            proc_cov=proc_scale * PROC_FLOOR,
        )
    else:
        ekf = TwoRangeESEKF(
            mdl.dynamics,
            obs2,
            mdl.MANIFOLD,
            in_cov=in_cov,
            obs_cov=np.diag(np.r_[RANGE_VAR, RANGE_VAR, np.full(3, ATT_VAR)]),
            proc_cov=proc_scale * PROC_FLOOR,
        )
    x_l1 = np.r_[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
    ]  # leader 1 abs (hover, level)
    x_f = np.r_[
        0.0, 2.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
    ]  # follower truth abs (~2 m, level)
    x_true = np.array(mdl.from_absolute_state(jnp.asarray(x_l1), jnp.asarray(x_f)))
    rad = x_true[0:3] / np.linalg.norm(x_true[0:3])
    that = np.cross(rad, [0.0, 0.0, 1.0])
    that /= np.linalg.norm(that)
    off = np.r_[
        offset_mag * that, np.zeros(6)
    ]  # tangent offset: tangential relative-position error
    x_hat = np.array(mdl.MANIFOLD.boxplus(jnp.asarray(x_true), jnp.asarray(off)))
    P = np.diag(np.r_[np.full(3, 4.0), np.full(3, 0.01), np.full(3, 0.25)])
    b_a = rng.normal(
        0, ACCEL_BIAS_STD
    )  # accelerometer z-bias (the model's scalar thrust channel)
    prev_u = None
    errs, nees, dists = [], [], []
    for _i in range(STEPS):
        guess = np.tile(np.r_[U_HOVER, U_HOVER], (WINDOW, 1))
        if prev_u is not None:
            guess[:, 4:8] = np.vstack([prev_u[1:, 4:8], prev_u[-1:, 4:8]])
        res = ctrl.solve(jnp.asarray(x_hat), jnp.asarray(guess))
        prev_u = res.u
        f_f, w_f = float(res.u[0, 4]), np.asarray(res.u[0, 5:8])
        x_f = np.array(qsim(jnp.asarray(x_f), jnp.asarray(np.r_[f_f, w_f]))[0])
        x_f[3:7] /= np.linalg.norm(x_f[3:7])
        drag_body = np.array(
            elmath.quaternion_rotate_point(
                elmath.quaternion_inverse(jnp.asarray(x_f[3:7])),
                jnp.asarray(-K_DRAG * x_f[7:10]),
            )
        )
        if driven == "imu":
            f_imu = (
                f_f + drag_body[2] + b_a + rng.normal(0, ACCEL_NOISE)
            )  # accel sees drag-z
            w_imu = w_f + rng.normal(0, GYRO_NOISE, 3)
        else:
            f_imu, w_imu = f_f, w_f  # command-driven (#8 reproduction): blind to drag
        u_imu = np.r_[U_HOVER, f_imu, w_imu]
        x_hat, P = ekf.predict(
            jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(u_imu), DT
        )
        x_true = np.array(mdl.from_absolute_state(jnp.asarray(x_l1), jnp.asarray(x_f)))
        obs_fn = obs_lean if lean else obs2
        y = np.array(obs_fn(jnp.asarray(x_true)))
        y[0:2] += rng.normal(0, np.sqrt(RANGE_VAR), 2)
        y[2:6] = _att_noise(x_true[3:7], rng)
        if lean:  # IMU-derived relative-velocity measurement (stand-in: truth + noise)
            y[6:9] = x_true[7:10] + rng.normal(0, np.sqrt(VEL_VAR), 3)
        x_hat, P = ekf.update(jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(y))
        x_hat, P = np.array(x_hat), np.array(P)
        x_hat[3:7] /= np.linalg.norm(x_hat[3:7])
        p_f_hat = np.array(
            mdl.to_absolute_state(jnp.asarray(x_l1), jnp.asarray(x_hat))
        )[10:13]
        e = np.array(ekf._boxminus(jnp.asarray(x_hat), jnp.asarray(x_true)))
        errs.append(float(np.linalg.norm(p_f_hat - x_f[0:3])))
        nees.append(
            float(e[0:3] @ np.linalg.solve(P[0:3, 0:3] + 1e-9 * np.eye(3), e[0:3]))
        )
        dists.append(float(np.linalg.norm(x_true[0:3])))
    return np.array(errs), np.array(nees), np.array(dists)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=ORDER)
    ap.add_argument(
        "--closed", action="store_true", help="step 2: the carried-ESEKF closed corner"
    )
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument(
        "--proc-scale",
        type=float,
        default=1.0,
        help="scale the ESEKF process-noise floor",
    )
    ap.add_argument("--driven", default="both", choices=["both", "imu", "cmd"])
    ap.add_argument(
        "--rate-scale",
        type=float,
        default=1.0,
        help="scale follower body-rate bounds (gentler)",
    )
    args = ap.parse_args()
    if args.closed:
        ctrl, _ = build_o1_oac(
            two_range=True, order=args.order, rate_scale=args.rate_scale
        )
        print(
            "=== (E1,O1) CORNER step 2: O1 OA on a carried IMU-driven 2-range ESEKF, 2 leaders, closed ==="
        )
        print(
            f"{STEPS} steps, {args.seeds} seeds, order {args.order}; tangential init 2.0 m; bounded = "
            f"recovery < 1.0 m AND formation in [0.5, 6] m\n"
        )
        print(
            f"{'config':>26} {'rec_med':>8} {'rec_p90':>8} {'NEES':>7} {'%bnd':>6} {'distMed':>8}"
        )
        # (label, driven, lean): the lean estimator (velocity+attitude IMU/AHRS-measured) is the redirect
        cfgs = [
            ("O1, LEAN est (IMU vel+att)", "imu", True),
            ("O1, full ESEKF (range-only)", "imu", False),
            ("O1, full ESEKF, command (#8)", "cmd", False),
        ]
        if args.driven != "both":
            cfgs = [c for c in cfgs if c[1] == args.driven]
        for label, driven, lean in cfgs:
            rows = []
            for s in range(args.seeds):
                errs, nees, dists = closed_loop(
                    ctrl, s, driven=driven, proc_scale=args.proc_scale, lean=lean
                )
                bounded = errs[-1] < 1.0 and np.min(dists) > 0.5 and np.max(dists) < 6.0
                rows.append((errs[-1], np.median(nees), bounded, np.median(dists)))
            r = np.array(rows)
            print(
                f"{label:>26} {np.median(r[:, 0]):>8.2f} {np.percentile(r[:, 0], 90):>8.2f} "
                f"{np.median(r[:, 1]):>7.1f} {100 * np.mean(r[:, 2]):>5.0f}% {np.median(r[:, 3]):>8.2f}"
            )
        print(
            "\nVERDICT: does the (E1,O1) corner with 2 leaders stay BOUNDED + consistent (the carried full-"
            "quad loop the old approach diverged on)? IMU-driven vs command-driven (#8) is the E1 contrast."
        )
        return
    print(
        f"=== (E1,O1) corner step 1: 2-range full-quad OA, plan-on-truth (order {args.order}, 2 leaders) ==="
    )
    print(
        f"{STEPS} steps, window {WINDOW}, dist band [1,3] m; does the O1 OA solve + hold formation?\n"
    )
    print(
        f"{'config':>22} {'dist start':>11} {'dist end':>9} {'dist min':>9} {'dist max':>9} {'ms/solve':>9}"
    )
    for label, oac in [("no-OAC (hover-hold)", False), ("O1 OA, 2-range", True)]:
        ctrl, _ = build_o1_oac(two_range=True, order=args.order)
        d, wall = run(ctrl, oac=oac)
        print(
            f"{label:>22} {np.linalg.norm(X0[0:3]):>11.2f} {d[-1]:>9.2f} {d.min():>9.2f} {d.max():>9.2f} {wall:>9.1f}"
        )
    print(
        "\nREAD: the O1 OA should SOLVE (order-5 manifold STLOG, 2 ranges), keep the formation in the "
        "[1,3] m band (the dist constraint binds), and run at a sane solve time -- validating the corner's "
        "OA machinery before the carried ESEKF (step 2). A drive-away or solve failure flags a problem."
    )


if __name__ == "__main__":
    main()
