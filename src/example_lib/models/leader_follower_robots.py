"""
Copyright © 2024 H S Helson Go and Ching Lok Chong.

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import functools

import jax
import jax.numpy as jnp

from . import MODELS, Model, simple_robot
from .simple_robot import ObservationKind

NUM_STATES = 3
NUM_INPUTS = 2


@jax.jit
def dynamics(x, u, p=None):
    x = jnp.reshape(x, (-1, NUM_STATES))
    u = jnp.reshape(u, (-1, NUM_INPUTS))

    return jax.vmap(simple_robot.dynamics)(x, u).ravel()


@functools.partial(jax.jit, static_argnames=["kind"])
def observation(x, _, kind=ObservationKind.RANGE):
    x = jnp.reshape(x, (-1, NUM_STATES))
    leader_pos = x[0, 0:2]
    hdg = x[:, 2]

    relative_positions = leader_pos - x[1:, 0:2]

    if kind == ObservationKind.RANGE:
        relative_ranges = jnp.linalg.norm(relative_positions, axis=1)
        return jnp.concatenate([leader_pos, hdg, relative_ranges])

    else:
        raise ValueError(f"Invalid sensor kind {kind}")


def interrobot_distance(x0, us, cost):
    xs, _ = cost.eval_integrator(x0, us)

    xs = jnp.reshape(xs, (len(xs), -1, NUM_STATES))
    leader_pos = xs[:, [0], 0:2]
    follower_pos = xs[:, 1:, 0:2]
    return jnp.linalg.norm(follower_pos - leader_pos, axis=2).ravel()


MODELS["leader_follower_unicycles"] = Model(dynamics=dynamics, observation=observation)
