import equinox as eqx
import jax
import jax.numpy as jnp

from example_lib.math import rotation
from example_lib.models import MODELS, Model, quadrotor

from .simple_robot import ObservationKind

NUM_STATES = 10
NUM_INPUTS = 4


@jax.jit
def dynamics(x, u):
    x = jnp.reshape(x, (-1, NUM_STATES))
    u = jnp.reshape(u, (-1, NUM_INPUTS))

    return jax.vmap(quadrotor.dynamics)(x, u).ravel()


def extrinsics(tracker_state, target_state):
    tracker_position = tracker_state[0:3]
    tracker_attitude = tracker_state[3:7]
    target_position = target_state[0:3]
    return rotation.quaternion_rotate_point(
        tracker_attitude, target_position - tracker_position, invert_rotation=True
    )


def _distance(point):
    return jnp.linalg.norm(point)


@eqx.filter_jit
def observation(x, u, kind=ObservationKind.RANGE):
    if kind == ObservationKind.RANGE:
        sensor = _distance
    else:
        raise ValueError(f"Invalid sensor kind {kind}")
    x = jnp.reshape(x, (-1, 10))
    pos_ref = x[0, 0:3]
    att = x[:, 3:7].ravel()

    obs = jax.vmap(lambda x, y: sensor(extrinsics(x, y)), in_axes=(0, None))

    h_inter_robot = obs(x[1:, :], pos_ref).ravel()

    vel = x[:, 7:10].ravel()
    return jnp.concatenate([pos_ref, att, h_inter_robot, vel])


def interrobot_distance_squared(x0, us, cost):
    xs, _ = cost.eval_integrator(x0, us)

    xs = jnp.reshape(xs, (len(xs), -1, NUM_STATES))
    leader_pos = xs[:, [0], 0:3]
    follower_pos = xs[:, 1:, 0:3]

    return jnp.sum((follower_pos - leader_pos) ** 2, axis=2).ravel()


MODELS["leader_follower_quadrotors"] = Model(dynamics=dynamics, observation=observation)
