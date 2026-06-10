r"""Freeze-yaw test: in the ORIGINAL quad OA (range + relative-attitude obs, body-frame relative pose),
the optimizer commands heavy yaw (RMS 4.0, saturating). Does that yaw improve the POSITION observability,
or only the ATTITUDE block (-> an artifact for localizing the follower)?

Method: generate the OA orbit two ways -- yaw FREE (original bound) vs yaw FROZEN (omega_z ~ 0) --
planning on truth, and propagate the RECURSIVE POSTERIOR COVARIANCE (the reanchored EKF P-envelope =
the CRLB) along each. Compare trace(P) of the position / attitude / velocity blocks + the STLOG
objective.

VERIFIED RESULT (adversarial workflow): the ONLY robust conclusion is that the STLOG objective is
nearly FLAT in yaw (frozen vs free < 0.2% at every horizon) -- the OA optimizer is near-indifferent to
yaw, consistent with the range's yaw-symmetry. The position-covariance comparison is CONFOUNDED and
NOT a clean answer: freezing yaw makes the optimizer re-plan a DIFFERENT orbit (it pins the standoff to
the 1 m min-distance bound vs 3 m for free -- closer => better range^2 observability), and the position
ratio REVERSES with horizon (frozen/free 0.46x @40 steps -> 0.57x @80 -> 1.96x @120, frozen WORSE). The
same-orbit counterfactual (free orbit, yaw zeroed at fixed inputs) goes the other way: P_pos UP 1.33x +
standoff blows out to 10.8 m. So yaw is objective-flat but DYNAMICALLY ENTANGLED -- its direct effect on
localization is not separable from the standoff/orbit the optimizer chooses. In a strictly RANGE-ONLY
world (the JGCD paper), yaw is a symmetry of the objective => inert; the only way to make it matter is a
yaw-sensitive (bearing/FOV) sensor, which is a CONFOUND vs the range-only paper (and stripped from the
OG repo) -- so yaw stays a free flat output and the only OA lever is translation.

    PYTHONPATH=src JAX_PLATFORMS=cpu uv run python experiments/freeze_yaw_test.py --steps 60
"""

import argparse

from example_lib.models import inter_quadrotor_pose as mdl
import jax.numpy as jnp
import numpy as np
from observability_aware_control import integrator

from rt_oac import metrics
from rt_oac.controller import RTController
from rt_oac.error_state_ekf import ErrorStateEKF
from rt_oac.scenario import build_scenario
from rt_oac.warmstart import warm_guess

P0 = np.diag([
    2.0,
    2.0,
    2.0,
    1e-2,
    1e-2,
    1e-2,
    0.1,
    0.1,
    0.1,
])  # 9-tangent: big position uncertainty
INPUT_VAR = np.tile([0.05, 0.01, 0.01, 0.01], 2)
PROC_VAR = np.array([0.02, 0.02, 0.02, 1e-4, 1e-4, 1e-4, 0.05, 0.05, 0.05])


def renorm(x):
    x = np.array(x)
    x[3:7] /= np.linalg.norm(x[3:7])
    return x


def build_ctrl(sc, lo, hi, *, freeze):
    lb = np.array(sc.cfg["optim"]["lb"], float).copy()
    ub = np.array(sc.cfg["optim"]["ub"], float).copy()
    if freeze:  # pin the follower yaw-rate (omega_z) ~ 0; else keep the original asymmetric bound
        lb[3], ub[3] = -1e-3, 1e-3
    return RTController(
        sc.cost,
        stlog_dt=sc.stlog_dt,
        lb=lb,
        ub=ub,
        n_inputs=8,
        follower_indices=(4, 5, 6, 7),
        method="SLSQP",
        maxiter=6,
        constraint=mdl.interrobot_distance_squared,
        constraint_bounds=(lo**2, hi**2),
        constraint_mode="hard",
    )


def run_orbit(sc, ctrl, steps):
    """Plan the OA orbit on truth; propagate the recursive posterior covariance (reanchored EKF
    P-envelope). Returns per-step (trace P_pos, P_att, P_vel) and the mean follower yaw-rate + objective."""
    dt = float(sc.cfg["sim"]["integrator_dt"])
    res_var = np.r_[
        sc.cfg["noise"]["range_var"], np.full(3, sc.cfg["noise"]["att_var"])
    ]
    ekf = ErrorStateEKF(
        mdl.dynamics,
        lambda x: mdl.observation(x),
        mdl.MANIFOLD,
        in_cov=np.diag(INPUT_VAR),
        obs_cov=np.diag(res_var),
        proc_cov=np.diag(PROC_VAR),
        method=integrator.Methods.EULER,
    )
    x = np.asarray(sc.x0, float).copy()
    P = P0.copy()
    prev_u, tr_pos, tr_att, tr_vel, yawr, objs = None, [], [], [], [], []
    for i in range(steps):
        res = ctrl.solve(
            jnp.asarray(x), jnp.asarray(warm_guess(sc.reference_guess(i), prev_u))
        )
        prev_u = res.u
        u = np.asarray(res.u[0])
        yawr.append(abs(float(u[7])))  # follower yaw rate magnitude
        objs.append(float(res.objective))
        x_pred, P = ekf.predict(jnp.asarray(x), jnp.asarray(P), jnp.asarray(u), dt)
        x = renorm(x_pred)  # reanchor the mean to the (truth) prediction
        y = np.asarray(
            mdl.observation(jnp.asarray(x))
        )  # noiseless: the P update is innovation-free
        _, P = ekf.update(jnp.asarray(x), jnp.asarray(P), jnp.asarray(y))
        P = np.array(P)
        tr_pos.append(float(np.trace(P[0:3, 0:3])))
        tr_att.append(float(np.trace(P[3:6, 3:6])))
        tr_vel.append(float(np.trace(P[6:9, 6:9])))
    return (
        np.array(tr_pos),
        np.array(tr_att),
        np.array(tr_vel),
        float(np.mean(yawr)),
        float(np.mean(objs)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()
    sc = build_scenario(gramian_metric=metrics.neg_softmin_eig, use_manifold=True)
    lo = float(sc.cfg["opc"]["min_inter_vehicle_distance"])
    hi = float(sc.cfg["opc"]["max_inter_vehicle_distance"])

    print(
        f"Freeze-yaw test: {args.steps} steps, softmin-eig STLOG, recursive posterior covariance "
        f"(CRLB envelope). P0_pos trace = {np.trace(P0[0:3, 0:3]):.1f}\n"
    )
    print(
        f"{'yaw mode':>10} {'mean|yawrate|':>13} {'STLOG obj':>10} | final trace(P): "
        f"{'POSITION':>9} {'attitude':>9} {'velocity':>9}"
    )
    res = {}
    for name, fr in [("free", False), ("frozen", True)]:
        tp, ta, tv, yr, obj = run_orbit(
            sc, build_ctrl(sc, lo, hi, freeze=fr), args.steps
        )
        res[name] = (tp, ta, tv)
        print(
            f"{name:>10} {yr:>13.3f} {obj:>10.3f} | "
            f"{'':>16} {tp[-1]:>9.3f} {ta[-1]:>9.4f} {tv[-1]:>9.4f}"
        )
    fp, ff = res["free"], res["frozen"]
    print(
        "\n=== Does yaw help the POSITION (the physical localization) or only the ATTITUDE? ==="
    )
    print(
        f"  POSITION trace(P):  free {fp[0][-1]:.3f} -> frozen {ff[0][-1]:.3f}  "
        f"(frozen/free = {ff[0][-1] / fp[0][-1]:.2f}x; >1 means yaw HELPED position)"
    )
    print(
        f"  ATTITUDE trace(P):  free {fp[1][-1]:.4f} -> frozen {ff[1][-1]:.4f}  "
        f"(frozen/free = {ff[1][-1] / fp[1][-1]:.2f}x; >1 means yaw HELPED attitude)"
    )
    print(
        f"  VELOCITY trace(P):  free {fp[2][-1]:.4f} -> frozen {ff[2][-1]:.4f}  "
        f"(frozen/free = {ff[2][-1] / fp[2][-1]:.2f}x)"
    )
    print(
        "\n=> CONFOUNDED: free vs frozen are DIFFERENT optimized orbits (different standoff), so the "
        "position ratio is not a clean yaw effect (it reverses with horizon). The robust read is the "
        "STLOG objective gap above (~flat in yaw). See the module docstring / PROGRESS section 9."
    )


if __name__ == "__main__":
    main()
