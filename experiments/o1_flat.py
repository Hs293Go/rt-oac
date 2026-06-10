r"""Plan in FLAT space, evaluate the FULL-FIDELITY (O1) STLOG -- the robust realization of (E1-lean, O1).

The direct thrust+rates O1 corner caps at ~50% (the aggressive control destabilizes the carried estimate,
§6.10). The robust path: keep the validated O0 control architecture (flat-output plan + geometric tracker
+ lean translation INS, 95% at O0) but SCORE the maneuver with the O1 objective (the full-quad relative-pose
STLOG, the value adder). The link is the differential-flatness map: the OA's decision variables are the
follower's flat-output velocities; we push them through the Mellinger-Kumar map (accel -> thrust + tilt
attitude -> body rates) to reconstruct the quad inputs, then evaluate the order-5 relative-pose STLOG.

The flatness map is PORTED (differentiable, JAX) from `minsnap_trajectories`'s vetted
`flat_output_to_quadrotor_trajectory`: half-vector tilt quaternion (smooth) + body rates from the analytic
z-derivative dz = -z x (z x jerk)/|z| (NOT a finite-diff of the quaternion -- the source of my earlier
NaN/non-PSD). yaw = 0 (the follower yaw is inert in range-only).

This file: (1) the ported flatness map; (2) the O1-on-flat objective; (3) a validation that the objective
responds to the plan (a maneuver scores more observable than hover) and that gradient ascent through the
flatness map grows a sane observability maneuver. The closed loop (clins flat+track+lean-INS) is next.

    PYTHONPATH=src JAX_PLATFORMS=cpu uv run python experiments/o1_flat.py
"""

from example_lib import math as elmath
from example_lib.models import inter_quadrotor_pose as mdl
import jax
import jax.numpy as jnp
import numpy as np
from observability_aware_control import observability_cost

from rt_oac import metrics

ORDER, STLOG_DT, WINDOW, DT, GRAV = 5, 0.2, 10, 0.1, 9.81
RANGE_VAR, ATT_VAR = 0.01, 0.01
OBS_IDX = (0, 1, 2, 7, 8, 9)  # position + velocity (O1 observability target)
D12_ABS = jnp.array([2.0, 0.0, 1.5])
VAR2 = np.r_[RANGE_VAR, RANGE_VAR, np.full(4, ATT_VAR)]
LEADER_V = jnp.array([0.0, 0.0, 0.0])  # leader hovers (level)
GVEC = jnp.array([0.0, 0.0, GRAV])


def obs2(x, u=None):
    r_lf, q_fl = x[0:3], x[3:7]
    r2 = r_lf + elmath.quaternion_rotate_point(q_fl, D12_ABS)
    return jnp.concatenate([jnp.array([jnp.dot(r_lf, r_lf), jnp.dot(r2, r2)]), q_fl])


_o1 = observability_cost.ObservabilityCost(
    mdl.dynamics,
    obs2,
    DT,
    gramian_kw={"order": ORDER, "var": VAR2, "manifold": mdl.MANIFOLD},
    gramian_metric=metrics.neg_softmin_eig,
    observed_indices=OBS_IDX,
)


def flatness_map(v_w, p_f0, q_l, p_l0):
    """Differentiable Mellinger-Kumar map (ported from minsnap_trajectories, yaw=0, mass 1): follower
    world-velocity plan v_w (W+2, 3) -> relative-pose x0 [r_lf, q_fl, v_lf] + quad inputs us (W, 8)."""
    # plan points are STLOG_DT apart (the STLOG integrates the inputs at STLOG_DT, not the control DT) --
    # so the finite-diff accel/jerk must use STLOG_DT, else the implied maneuver is scaled wrong
    vel = v_w[:-2]  # (W, 3)
    acc = (v_w[1:-1] - v_w[:-2]) / STLOG_DT
    jer = (v_w[2:] - 2.0 * v_w[1:-1] + v_w[:-2]) / STLOG_DT**2
    z = acc + GVEC
    z = z.at[:, 2].set(
        jnp.maximum(z[:, 2], 2.0)
    )  # thrust can't invert (guards the tilt_den sqrt -> no NaN)
    z_nrm = jnp.linalg.norm(z, axis=1, keepdims=True)
    zb = z / z_nrm  # body-z (thrust axis)
    dz = -jnp.cross(zb, jnp.cross(zb, jer)) / z_nrm  # analytic z-derivative (from jerk)
    thrust = jnp.sum(zb * (acc + GVEC), axis=1)  # (W,)
    tilt_den = jnp.sqrt(
        2.0 * (1.0 + zb[:, 2])
    )  # half-vector tilt quaternion (smooth except z_z=-1)
    q = jnp.stack(
        [
            -zb[:, 1] / tilt_den,
            zb[:, 0] / tilt_den,
            jnp.zeros(zb.shape[0]),
            0.5 * tilt_den,
        ],
        axis=1,
    )  # xyzw, yaw=0
    omg_den = zb[:, 2] + 1.0
    omg_term = dz[:, 2] / omg_den
    w_f = jnp.stack(
        [
            -dz[:, 1] + zb[:, 1] * omg_term,  # wx (yaw=0, yaw_rate=0)
            dz[:, 0] - zb[:, 0] * omg_term,  # wy
            (zb[:, 1] * dz[:, 0] - zb[:, 0] * dz[:, 1]) / omg_den,  # wz
        ],
        axis=1,
    )
    x_l = jnp.concatenate([p_l0, q_l, LEADER_V])
    x_f0 = jnp.concatenate([p_f0, q[0], vel[0]])
    x0_rel = mdl.from_absolute_state(x_l, x_f0)
    us = jnp.concatenate(
        [
            jnp.full((thrust.shape[0], 1), GRAV),
            jnp.zeros_like(w_f),
            thrust[:, None],
            w_f,
        ],
        axis=1,
    )  # [f_l, w_l(3), f_f, w_f(3)]
    return x0_rel, us


def o1_on_flat(v_w, p_f0, q_l, p_l0):
    """The O1 objective scored on a flat-output velocity plan (lower neg_softmin_eig = more observable)."""
    x0_rel, us = flatness_map(v_w, p_f0, q_l, p_l0)
    return _o1(x0_rel, us, STLOG_DT).objective


def main():
    p_l0 = jnp.array([0.0, 0.0, 0.0])
    q_l = jnp.array([0.0, 0.0, 0.0, 1.0])
    p_f0 = jnp.array([0.0, 2.0, 0.5])  # follower ~2 m from leader
    n = WINDOW + 2
    v_hover = jnp.zeros((n, 3))  # hover-hold plan (no maneuver)
    th = jnp.linspace(0, 2 * jnp.pi, n)
    v_man = jnp.stack(
        [1.2 * jnp.sin(th), 0.8 * jnp.cos(th), 0.4 * jnp.sin(2 * th)], axis=1
    )  # a maneuver

    j_hover = float(o1_on_flat(v_hover, p_f0, q_l, p_l0))
    j_man = float(o1_on_flat(v_man, p_f0, q_l, p_l0))
    print(
        "=== Plan-flat / evaluate-O1-STLOG (flatness map ported from minsnap_trajectories) ==="
    )
    print("O1 objective (neg_softmin_eig; LOWER = more observable):")
    print(f"  hover-hold plan : {j_hover:+.4e}")
    print(
        f"  maneuver plan   : {j_man:+.4e}   (should be MORE negative = more observable)\n"
    )

    grad = jax.jit(jax.grad(o1_on_flat))
    rng = np.random.default_rng(0)
    v = v_hover + 0.05 * jnp.asarray(
        rng.standard_normal((n, 3))
    )  # symmetry-break the hover critical point
    print(
        "BOUNDED gradient ascent on observability (clip |vel| <= 3 m/s -- the OA's real velocity bound):"
    )
    for i in range(30):
        g = grad(v, p_f0, q_l, p_l0)
        v = jnp.clip(
            v - 0.01 * g, -3.0, 3.0
        )  # descend neg_softmin_eig, clipped to the velocity bound
        if i % 5 == 4:
            print(
                f"  step {i + 1:>2}: O1 obj {float(o1_on_flat(v, p_f0, q_l, p_l0)):+.4e}, "
                f"mean |vel| {float(jnp.linalg.norm(v, axis=1).mean()):.2f} m/s"
            )
    print(
        "\nREAD: if the maneuver scores more observable than hover AND gradient ascent through the ported "
        "flatness map grows a maneuver that increases observability (more negative O1 obj), the "
        "differentiable plan-flat/score-O1 machinery works -- ready to drop into the clins loop."
    )


if __name__ == "__main__":
    main()
