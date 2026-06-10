r"""O1 observability probe: does HIGHER-ORDER STLOG reveal the full-quad velocity + tangential position?

The O0 probe (higher_order_obs.py) showed higher order does NOTHING for the flat model -- because there
velocity is a droppable INPUT. O1 is different and is the paper's justification for high-order STLOG: the
full relative-pose model `inter_quadrotor_pose` x = [r_lf(3), q_fl(4), v_lf(3)] integrates velocity as a
STATE (r_lf' = v_lf + r_lf x w; v_lf' = R(q_fl) t_l - t_f + ...), and the observation is [range^2, q_fl] --
the relative ATTITUDE is measured, but the relative VELOCITY and the 2 TANGENTIAL position coords are
observed ONLY through the range and its time-derivatives (the coupled dynamics chain pos<-vel<-accel<-att).
So they are structurally unobserved at low order and -- the paper's claim -- revealed by HIGHER order.

This probe builds the manifold STLOG (9x9 tangent [pos(3), att(3), vel(3)]) at orders 1-5 and reports the
tangential-position and velocity 1-sigma (from the STLOG pseudo-inverse) + the rank, for 1 leader (range +
attitude) and 2 leaders (range1 + range2 + attitude). If higher order SHRINKS the velocity/tangential
1-sigma (and/or lifts the rank) -- unlike O0 -- the paper's high-order justification holds for O1; and the
2nd leader should help structurally (a 2nd range channel).

    PYTHONPATH=src JAX_PLATFORMS=cpu uv run python experiments/o1_obs_probe.py
"""

from example_lib.models import inter_quadrotor_pose as mdl
import jax.numpy as jnp
import numpy as np
from observability_aware_control import observability_cost

RANGE_VAR, ATT_VAR, GRAV = 1e-2, 1e-2, 9.81
R_LF = np.array([5.0, 0.0, 0.0])  # relative position: radial = x; tangential = y, z
QFL = np.array([0.0, 0.0, 0.0, 1.0])  # identity relative attitude (xyzw)
V_LF = np.array([
    0.0,
    1.0,
    0.0,
])  # a tangential relative velocity (representative maneuver)
X = np.r_[R_LF, QFL, V_LF]
U = np.array([
    GRAV,
    0.0,
    0.0,
    0.0,
    GRAV,
    0.0,
    0.0,
    0.0,
])  # both agents hover (f=g, no body rate)
THAT = np.array([0.0, 1.0, 0.0])  # a tangential direction (perp to the radial x)
D12 = np.array([
    0.0,
    8.0,
    0.0,
])  # 2nd leader offset (relative frame): tangential placement
VAR1 = np.r_[RANGE_VAR, np.full(4, ATT_VAR)]  # [range^2, q_fl(4)]
VAR2 = np.r_[RANGE_VAR, RANGE_VAR, np.full(4, ATT_VAR)]  # [range1^2, range2^2, q_fl(4)]


def obs2(x, u=None):  # 2-leader observation: range1^2, range2^2, relative attitude
    r_lf, q_fl = x[0:3], x[3:7]
    r2 = r_lf - D12
    return jnp.concatenate([jnp.array([jnp.dot(r_lf, r_lf), jnp.dot(r2, r2)]), q_fl])


def probe(obs, var, dt):
    print(
        f"{'order':>6} {'tang-pos 1sig':>14} {'vel 1sig':>10} {'rank/9':>7} {'min-eig':>10}"
    )
    for order in (1, 2, 3, 4, 5):
        cost = observability_cost.ObservabilityCost(
            mdl.dynamics,
            obs,
            dt,
            gramian_kw={"order": order, "var": var, "manifold": mdl.MANIFOLD},
            gramian_metric=lambda g: g,
            observed_indices=(),
        )
        gm = np.asarray(cost.eval_gramian(jnp.asarray(X), jnp.asarray(U), dt))
        gm = 0.5 * (gm + gm.T)
        evals = np.linalg.eigvalsh(gm)
        # regularized inverse with a 3 m / 3 m/s prior: observable dirs -> small sigma, unobservable -> ~prior
        prior = np.diag(
            1.0 / np.r_[np.full(3, 3.0**2), np.full(3, 1.0**2), np.full(3, 3.0**2)]
        )
        cov = np.linalg.inv(gm + prior)
        sig_pos_t = np.sqrt(max(THAT @ cov[0:3, 0:3] @ THAT, 0.0))
        sig_vel = np.sqrt(max(np.trace(cov[6:9, 6:9]) / 3.0, 0.0))
        rank = int(np.sum(evals > 1e-9 * max(evals.max(), 1e-30)))
        print(
            f"{order:>6} {sig_pos_t:>14.3f} {sig_vel:>10.3f} {rank:>5}/9 {evals[0]:>10.2e}"
        )


def main():
    dt = 0.2
    print(
        "=== O1 observability probe (full relative-pose quad, manifold STLOG, dt=0.2) ==="
    )
    print(
        "(does higher order reveal the velocity + tangential position? lower 1sig / higher rank = yes)\n"
    )
    print("--- 1 leader: obs = [range^2, relative-attitude] ---")
    probe(mdl.observation, VAR1, dt)
    print(
        "\n--- 2 leaders: obs = [range1^2, range2^2, relative-attitude] (2nd leader offset tangentially) ---"
    )
    probe(obs2, VAR2, dt)
    print(
        "\nREAD: O0 (higher_order_obs.py) was FLAT across orders -- velocity is a droppable input there. If "
        "O1 here SHRINKS the velocity/tangential 1sig (or lifts rank) with order, the paper's high-order "
        "justification holds for the full quad. 2 leaders should help structurally (2nd range channel)."
    )


if __name__ == "__main__":
    main()
