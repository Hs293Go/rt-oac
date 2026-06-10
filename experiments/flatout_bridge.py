r"""Rung 3a BRIDGE: is the OA flat-output trajectory actually quad-realizable?

The premise (differential flatness): an OA trajectory planned in the flat-output space
sigma=[x,y,z,psi] (experiments/flatout_oac_validate.py) is trackable by a real quadrotor via the
subservient geometric tracker -- "as long as the tracking controller tracks well". This closes that
loop end to end:

  OA flat-output reference sigma(t)  ->  TrackingController (pos+yaw -> thrust+attitude)
       -> AttitudeController (attitude -> body-rate)  ->  10-state quadrotor.dynamics

Tests: (1) TRACKABILITY -- does the quad follow the OA reference with small position error? (2) OA-
BENEFIT TRANSFER -- does a range-only EKF run on the quad's ACTUAL trajectory still recover the
unobservable (tangential) follower-position error, the way the kinematic planner did? Compares the OA
reference against a no-OAC straight-line reference.

    PYTHONPATH=src JAX_PLATFORMS=cpu uv run python experiments/flatout_bridge.py --ref /tmp/oa_ref.npz
"""

import argparse

from example_lib.misc import simple_ekf
from example_lib.misc.geometric_controller import (
    AttitudeController,
    TrackingController,
)
from example_lib.models import quadrotor
import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation
import numpy as np
from observability_aware_control import integrator


def yaw_quat(psi):
    return np.array(Rotation.from_euler("z", float(psi)).as_quat())  # xyzw


def track_on_quad(ref, dt, n_sub=10):
    """Track a flat-output reference [pos(3), psi, vel(3), leader_pos(3)] on the 10-state quad via
    TrackingController + AttitudeController, with the inner loop run n_sub times faster than the OA
    step (realistic cascade) and the reference linearly interpolated across the fast steps (so the
    coarse-OA velocity jumps are smoothed). Returns the quad's actual (pos, vel) trajectory at OA rate."""
    dti = dt / n_sub
    tctrl, actrl = TrackingController(), AttitudeController()
    qdyn = jax.jit(
        integrator.Integrator(quadrotor.dynamics, integrator.Methods.RK4, stepsize=dti)
    )
    x = np.r_[
        ref[0, 0:3], yaw_quat(ref[0, 3]), ref[0, 4:7]
    ]  # [pos(3), quat(4 xyzw), vel(3)]
    traj = []
    for i in range(len(ref)):
        p0, v0, y0 = ref[i, 0:3], ref[i, 4:7], ref[i, 3]
        j = min(i + 1, len(ref) - 1)
        p1, v1, y1 = ref[j, 0:3], ref[j, 4:7], ref[j, 3]
        acc = (v1 - v0) / dt
        for k in range(n_sub):
            a = (k + 0.5) / n_sub  # interpolation fraction across the OA step
            rp, rv, ry = (
                (1 - a) * p0 + a * p1,
                (1 - a) * v0 + a * v1,
                (1 - a) * y0 + a * y1,
            )
            st = TrackingController.State(
                jnp.asarray(x[0:3]), jnp.asarray(x[3:7]), jnp.asarray(x[7:10])
            )
            rf = TrackingController.Reference(
                jnp.asarray(rp), jnp.asarray(rv), jnp.asarray(acc), float(ry)
            )
            out, _ = tctrl.run(st, rf)
            ar, _ = actrl.run(
                AttitudeController.State(jnp.asarray(x[3:7])),
                AttitudeController.Reference(out.orientation),
            )
            u = np.r_[float(out.thrust), np.asarray(ar.body_rate)]
            x = np.array(qdyn(jnp.asarray(x), jnp.asarray(u))[0])
            x[3:7] /= np.linalg.norm(x[3:7])
        traj.append(x.copy())
    traj = np.array(traj)
    return traj[:, 0:3], traj[:, 7:10]


# --- range-only EKF on the RELATIVE position r = follower - leader (so obs |r| is a fixed function,
#     no per-step leader closure); predict with the relative velocity (ref command - leader velocity) ---
def _rel_dyn(x, u, dt):
    return x + dt * u  # relative-position single-integrator


def _rel_obs(x):
    return jnp.array([jnp.linalg.norm(x)])  # range = |r|


_REL_EKF = simple_ekf.SimpleEKF(
    _rel_dyn, _rel_obs, in_cov=np.diag([1e-2, 1e-2, 1e-2]), obs_cov=np.diag([1e-2])
)


def ekf_recover(actual_pos, ref_vel, leader_pos, leader_vel, offset, dt, seed):
    """Range-only EKF on the follower's ACTUAL trajectory (relative-position form), predicting with the
    reference relative-velocity command. Returns (final, RMSE) of |estimate - true relative position|."""
    rng = np.random.default_rng(seed)
    range_var = 1e-2
    r_true = actual_pos - leader_pos
    r_hat = r_true[0] + offset  # the unobservable initial follower-position error
    P = np.diag([2.0, 2.0, 2.0])
    errs = []
    for i in range(len(actual_pos)):
        u = ref_vel[i] - leader_vel[i]  # relative velocity command
        r_hat, P = _REL_EKF.predict(
            jnp.asarray(r_hat), jnp.asarray(P), jnp.asarray(u), dt
        )
        y = np.array([np.linalg.norm(r_true[i])]) + rng.normal(0, np.sqrt(range_var))
        r_hat, P = _REL_EKF.update(jnp.asarray(r_hat), jnp.asarray(P), jnp.asarray(y))
        r_hat, P = np.array(r_hat), np.array(P)
        errs.append(float(np.linalg.norm(r_hat - r_true[i])))
    return errs[-1], float(np.sqrt(np.mean(np.square(errs))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ref",
        default="/tmp/oa_ref.npz",  # noqa: S108
        help="OA flat-output reference from flatout_oac_validate.py --dump",
    )
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    data = np.load(args.ref)
    ref_dt = float(data["dt"])
    oa = data["ref"]  # (T, 10): follower [x,y,z,psi], vel(3), leader_pos(3)
    leader_pos = oa[:, 7:10]

    # tangential-horizontal initial error (the unobservable direction), magnitude 2.26 m
    r = oa[0, 0:3] - leader_pos[0]
    rhat = r / np.linalg.norm(r)
    that_h = np.cross(rhat, np.array([0.0, 0.0, 1.0]))
    that_h /= np.linalg.norm(that_h)
    err0 = 2.26 * that_h

    print(
        "=== Rung 3a bridge: track the OA flat-output reference on the 10-state quad ==="
    )
    print(f"reference: {oa.shape[0]} steps, dt={ref_dt}\n")

    # --- track the OA reference and a NO-OAC straight reference on the quad ---
    oa_pos, oa_vel = track_on_quad(oa, ref_dt)
    # no-OAC: follower flies straight, matching the leader's velocity (constant offset), yaw 0
    straight = oa.copy()
    lead_vel = np.gradient(leader_pos, ref_dt, axis=0)
    straight[:, 0:3] = oa[0, 0:3] + (leader_pos - leader_pos[0])  # rigid follow
    straight[:, 3] = 0.0
    straight[:, 4:7] = lead_vel
    st_pos, st_vel = track_on_quad(straight, ref_dt)

    for label, ref_full, qpos, _qvel in [
        ("OA (maneuver)", oa, oa_pos, oa_vel),
        ("no-OAC (straight)", straight, st_pos, st_vel),
    ]:
        track_err = np.linalg.norm(qpos - ref_full[:, 0:3], axis=1)
        rec = np.array([
            ekf_recover(qpos, ref_full[:, 4:7], leader_pos, lead_vel, err0, ref_dt, s)
            for s in range(args.seeds)
        ])
        print(
            f"{label:>18}: track_err RMS {np.sqrt(np.mean(track_err**2)):.3f} m, max {track_err.max():.3f} m "
            f"| EKF recovery final {np.median(rec[:, 0]):.3f} m, RMSE {np.median(rec[:, 1]):.3f} m"
        )
    print(
        "\n(OA-benefit transfers iff the quad tracks the OA reference well AND its EKF recovery is "
        "much lower than the straight one's ~2.26 m.)"
    )


if __name__ == "__main__":
    main()
