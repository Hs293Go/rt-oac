r"""(E1,O1) VARIATION: a barometer on the follower instead of a 2nd known leader.

The deployable (E1,O0) corner used a 2nd KNOWN leader to remove the tangential ridge by geometry. A 2nd
GNSS anchor is expensive/unrealistic; a BAROMETER is a standard onboard sensor. Geometrically they are
comparable for the O1 corner: a single range to leader 1 leaves the TWO tangential (cross-range) directions
weak; a 2nd range removes one of them, a barometer removes a DIFFERENT one (the vertical). Both leave ONE
weak horizontal direction for the O1 observability-aware maneuver to resolve. So: does a cheap onboard
barometer substitute for the 2nd anchor in the (E1,O1) loop?

The barometer measures the follower's ABSOLUTE altitude, which (with the leader's known altitude) is an
exact, differentiable function of the relative-pose state: ``baro(x) = to_absolute_state(x_l1, x)[12]`` (the
follower's world z). It is added BOTH to the estimator's measurement and to the O1 OA's STLOG observation,
so the planner knows the vertical is sensed and focuses the maneuver on the remaining horizontal weak
direction.

Held identical to o1_corner's closed corner (carried IMU-driven relative-pose ESEKF, direct thrust+rates O1
OA, full quad truth + drag); only the OBSERVATION SET changes. Compares, on the lean estimator (E1):
  1 range            (the diverging floor)
  1 range + baro     (the variation -- deployable, no 2nd anchor)
  2 ranges           (the existing 2-leader reference)

    PYTHONPATH=src JAX_PLATFORMS=cpu uv run python experiments/baro_o1.py --seeds 20
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
import o1_corner as o1c

from example_lib import math as elmath
from example_lib.models import inter_quadrotor_pose as mdl
from observability_aware_control import integrator, observability_cost
from rt_oac import metrics
from rt_oac.controller import RTController
from rt_oac.error_state_ekf import ErrorStateEKF

X_L1 = jnp.array([
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
])  # leader 1 hovers level at origin
BARO_STD = 0.3  # barometer altitude noise (m); set by --baro-std
PROC_SCALE = 1.0  # scale the ESEKF process-noise floor (set by --proc-scale); use 1/n_sub to CONSERVE the
# per-OA-step process noise when sub-stepping the predict (isolates integration rate from proc-noise level)


def baro_z(
    x,
):  # follower world altitude from the relative estimate (exact, differentiable)
    return mdl.to_absolute_state(X_L1, x)[12]


def obs_baro(x, u=None):  # OA / estimator base obs: 1 range + attitude + barometer
    r_lf, q = x[0:3], x[3:7]
    return jnp.concatenate([
        jnp.array([jnp.dot(r_lf, r_lf)]),
        q,
        jnp.array([baro_z(x)]),
    ])


def obs_1r_lean(x, u=None):  # lean estimator: 1 range + attitude + (measured) velocity
    return jnp.concatenate([mdl.observation(x), x[7:10]])


def obs_baro_lean(
    x, u=None
):  # lean estimator: 1 range + attitude + barometer + velocity
    return jnp.concatenate([obs_baro(x), x[7:10]])


class OneRangeLeanESEKF(ErrorStateEKF):  # residual [range, att(3), vel(3)]
    def _residual(self, x, y):
        hx = self._observation(x)
        att = elmath.quaternion_log(
            elmath.quaternion_product(elmath.quaternion_inverse(hx[1:5]), y[1:5])
        )
        return jnp.concatenate([y[0:1] - hx[0:1], att, y[5:8] - hx[5:8]])


class BaroLeanESEKF(ErrorStateEKF):  # residual [range, att(3), baro, vel(3)]
    def _residual(self, x, y):
        hx = self._observation(x)
        att = elmath.quaternion_log(
            elmath.quaternion_product(elmath.quaternion_inverse(hx[1:5]), y[1:5])
        )
        return jnp.concatenate([
            y[0:1] - hx[0:1],
            att,
            y[5:6] - hx[5:6],
            y[6:9] - hx[6:9],
        ])


def build_baro_oac(maxiter=6):
    var = np.r_[o1c.RANGE_VAR, np.full(4, o1c.ATT_VAR), BARO_STD**2]
    cost = observability_cost.ObservabilityCost(
        mdl.dynamics,
        obs_baro,
        o1c.DT,
        gramian_kw={"order": o1c.ORDER, "var": var, "manifold": mdl.MANIFOLD},
        gramian_metric=metrics.neg_softmin_eig,
        observed_indices=o1c.OBS_IDX,
    )
    return RTController(
        cost,
        stlog_dt=o1c.STLOG_DT,
        lb=o1c.LB,
        ub=o1c.UB,
        n_inputs=8,
        follower_indices=(4, 5, 6, 7),
        method="SLSQP",
        maxiter=maxiter,
        constraint=mdl.interrobot_distance_squared,
        constraint_bounds=o1c.DIST2,
    )


def _lean_ekf(cls, obs_fn, extra_obs_var):
    in_cov = np.diag(
        np.r_[1e-4, np.full(3, 1e-4), o1c.ACCEL_NOISE**2, np.full(3, o1c.GYRO_NOISE**2)]
    )
    obs_cov = np.diag(
        np.r_[
            o1c.RANGE_VAR,
            np.full(3, o1c.ATT_VAR),
            extra_obs_var,
            np.full(3, o1c.VEL_VAR),
        ]
    )
    return cls(
        mdl.dynamics,
        obs_fn,
        mdl.MANIFOLD,
        in_cov=in_cov,
        obs_cov=obs_cov,
        proc_cov=PROC_SCALE * o1c.PROC_FLOOR,
    )


def _corrupt(y, x_true, rng, *, n_range, baro):
    y = y.copy()
    y[0:n_range] += rng.normal(0, np.sqrt(o1c.RANGE_VAR), n_range)
    y[n_range : n_range + 4] = o1c._att_noise(x_true[3:7], rng)
    j = n_range + 4
    if baro:
        y[j] += rng.normal(0, BARO_STD)
        j += 1
    y[j : j + 3] = x_true[7:10] + rng.normal(
        0, np.sqrt(o1c.VEL_VAR), 3
    )  # lean: measured velocity
    return y


CONFIGS = {
    "1range": {
        "build_oa": lambda mi: o1c.build_o1_oac(two_range=False, maxiter=mi)[0],
        "ekf": lambda: _lean_ekf(OneRangeLeanESEKF, obs_1r_lean, []),
        "obs": obs_1r_lean,
        "corrupt": lambda y, xt, rng: _corrupt(y, xt, rng, n_range=1, baro=False),
    },
    "1range+baro": {
        "build_oa": build_baro_oac,
        "ekf": lambda: _lean_ekf(BaroLeanESEKF, obs_baro_lean, [BARO_STD**2]),
        "obs": obs_baro_lean,
        "corrupt": lambda y, xt, rng: _corrupt(y, xt, rng, n_range=1, baro=True),
    },
    "2range": {
        "build_oa": lambda mi: o1c.build_o1_oac(two_range=True, maxiter=mi)[0],
        "ekf": lambda: o1c.LeanESEKF(
            mdl.dynamics,
            o1c.obs_lean,
            mdl.MANIFOLD,
            in_cov=np.diag(
                np.r_[
                    1e-4,
                    np.full(3, 1e-4),
                    o1c.ACCEL_NOISE**2,
                    np.full(3, o1c.GYRO_NOISE**2),
                ]
            ),
            obs_cov=np.diag(
                np.r_[
                    o1c.RANGE_VAR,
                    o1c.RANGE_VAR,
                    np.full(3, o1c.ATT_VAR),
                    np.full(3, o1c.VEL_VAR),
                ]
            ),
            proc_cov=PROC_SCALE * o1c.PROC_FLOOR,
        ),
        "obs": o1c.obs_lean,
        "corrupt": lambda y, xt, rng: _corrupt(y, xt, rng, n_range=2, baro=False),
    },
}


def closed_loop(cfg, ctrl, seed, *, offset_mag=2.0, n_sub=1, meas_every=1):
    """o1_corner.closed_loop's (E1,O1) carried-ESEKF loop, observation set parameterized by cfg. n_sub =
    truth + IMU + EKF-predict sub-steps per OA replan (sets the PREDICT integration rate); meas_every = fuse
    the range/baro every meas_every sub-steps (sets the MEASUREMENT rate). (n_sub=5, meas_every=1) = both at
    100 Hz; (n_sub=5, meas_every=5) = predict 100 Hz but measurements 20 Hz -- isolates measurement rate from
    predict-integration rate (OA stays 20 Hz). (n_sub=1, meas_every=1) = the baseline."""
    rng = np.random.default_rng(seed)
    qsim = jax.jit(
        integrator.Integrator(
            o1c._quad_drag, integrator.Methods.RK4, stepsize=o1c.DT / n_sub
        )
    )
    ekf = cfg["ekf"]()
    obs_fn, corrupt = cfg["obs"], cfg["corrupt"]

    x_l1 = np.r_[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    x_f = np.r_[0.0, 2.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    x_true = np.array(mdl.from_absolute_state(jnp.asarray(x_l1), jnp.asarray(x_f)))
    rad = x_true[0:3] / np.linalg.norm(x_true[0:3])
    that = np.cross(rad, [0.0, 0.0, 1.0])
    that /= np.linalg.norm(that)
    off = np.r_[
        offset_mag * that, np.zeros(6)
    ]  # tangential (unobservable) initial position error
    x_hat = np.array(mdl.MANIFOLD.boxplus(jnp.asarray(x_true), jnp.asarray(off)))
    P = np.diag(np.r_[np.full(3, 4.0), np.full(3, 0.01), np.full(3, 0.25)])
    b_a = rng.normal(0, o1c.ACCEL_BIAS_STD)
    prev_u = None
    errs, nees, dists = [], [], []
    for _i in range(o1c.STEPS):
        guess = np.tile(np.r_[o1c.U_HOVER, o1c.U_HOVER], (o1c.WINDOW, 1))
        if prev_u is not None:
            guess[:, 4:8] = np.vstack([prev_u[1:, 4:8], prev_u[-1:, 4:8]])
        res = ctrl.solve(jnp.asarray(x_hat), jnp.asarray(guess))
        prev_u = res.u
        f_f, w_f = float(res.u[0, 4]), np.asarray(res.u[0, 5:8])
        for m in range(
            n_sub
        ):  # hold the OA command; truth + IMU + predict every sub-step
            x_f = np.array(qsim(jnp.asarray(x_f), jnp.asarray(np.r_[f_f, w_f]))[0])
            x_f[3:7] /= np.linalg.norm(x_f[3:7])
            drag_body = np.array(
                elmath.quaternion_rotate_point(
                    elmath.quaternion_inverse(jnp.asarray(x_f[3:7])),
                    jnp.asarray(-o1c.K_DRAG * x_f[7:10]),
                )
            )
            f_imu = f_f + drag_body[2] + b_a + rng.normal(0, o1c.ACCEL_NOISE)
            w_imu = w_f + rng.normal(0, o1c.GYRO_NOISE, 3)
            u_imu = np.r_[o1c.U_HOVER, f_imu, w_imu]
            x_hat, P = ekf.predict(
                jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(u_imu), o1c.DT / n_sub
            )
            x_hat, P = np.array(x_hat), np.array(P)
            x_true = np.array(
                mdl.from_absolute_state(jnp.asarray(x_l1), jnp.asarray(x_f))
            )
            if (m + 1) % meas_every == 0:  # fuse range/baro at the measurement rate
                y = corrupt(np.array(obs_fn(jnp.asarray(x_true))), x_true, rng)
                x_hat, P = ekf.update(
                    jnp.asarray(x_hat), jnp.asarray(P), jnp.asarray(y)
                )
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
    global BARO_STD, PROC_SCALE  # noqa: PLW0603 (CLI-set module knobs)
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument(
        "--baro-std", type=float, default=0.3, help="barometer altitude noise std [m]"
    )
    ap.add_argument("--only", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument(
        "--predict-hz",
        type=float,
        default=20.0,
        help="EKF predict integration rate [Hz] (truth + predict sub-steps). 20=baseline, 100=fine",
    )
    ap.add_argument(
        "--meas-hz",
        type=float,
        default=20.0,
        help="range/baro fusion rate [Hz]; OA stays 20 Hz. Set < predict-hz to isolate the measurement rate",
    )
    ap.add_argument(
        "--proc-scale",
        type=float,
        default=1.0,
        help="scale the ESEKF process-noise floor; pass 1/(predict_hz/20) to CONSERVE total proc noise",
    )
    args = ap.parse_args()
    BARO_STD = args.baro_std
    PROC_SCALE = args.proc_scale
    n_sub = max(
        1, round(args.predict_hz * o1c.DT)
    )  # predict sub-steps per OA replan (DT=0.05 -> 20 Hz base)
    meas_every = max(
        1, round(args.predict_hz / args.meas_hz)
    )  # fuse every meas_every sub-steps
    print(
        "=== (E1,O1) VARIATION: barometer instead of a 2nd leader (lean estimator, direct thrust+rates) ==="
    )
    print(
        f"{o1c.STEPS} steps, {args.seeds} seeds, baro_std={BARO_STD} m, predict={args.predict_hz:.0f} Hz, "
        f"meas={args.meas_hz:.0f} Hz (OA 20 Hz); tangential init 2.0 m; bounded = recovery<1.0 m AND "
        "formation in [0.5,6] m.\n"
    )
    print(
        f"{'config':>16} {'rec_med':>8} {'rec_p90':>8} {'NEES':>7} {'%bnd':>6} {'distMed':>8}"
    )
    for name in args.only:
        cfg = CONFIGS[name]
        ctrl = cfg["build_oa"](6)
        rows = []
        for s in range(args.seeds):
            errs, nees, dists = closed_loop(
                cfg, ctrl, s, n_sub=n_sub, meas_every=meas_every
            )
            bounded = errs[-1] < 1.0 and np.min(dists) > 0.5 and np.max(dists) < 6.0
            rows.append((errs[-1], np.median(nees), bounded, np.median(dists)))
        r = np.array(rows)
        print(
            f"{name:>16} {np.median(r[:, 0]):>8.2f} {np.percentile(r[:, 0], 90):>8.2f} "
            f"{np.median(r[:, 1]):>7.1f} {100 * np.mean(r[:, 2]):>5.0f}% {np.median(r[:, 3]):>8.2f}"
        )
    print(
        "\nVERDICT: does '1range+baro' (a cheap onboard barometer, NO 2nd anchor) match '2range' -- i.e. "
        "can a barometer substitute for the 2nd known leader in the (E1,O1) loop? '1range' is the floor."
    )


if __name__ == "__main__":
    main()
