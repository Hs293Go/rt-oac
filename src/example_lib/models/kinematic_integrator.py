"""
Copyright © 2025 Hs293Go.

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

import jax
import jax.numpy as jnp

from example_lib.math.rotation import angle_rotate_point
from example_lib.models import MODELS, Model


@jax.jit
def individual_dynamics(x: jax.Array, u: jax.Array):
    phi = x[3]

    v = u[0:3]
    w = u[3]
    dp = jnp.append(angle_rotate_point(phi, v[0:2]), v[2])
    return jnp.concatenate([dp, jnp.array([w])])


@jax.jit
def dynamics(x: jax.Array, u: jax.Array):
    num_individuals = x.shape[0] // 4
    x = jnp.reshape(x, (num_individuals, 4))
    u = jnp.reshape(u, (num_individuals, 4))

    return jax.vmap(individual_dynamics)(x, u).ravel()


@jax.jit
def observation(x: jax.Array, _: jax.Array | None = None):
    num_individuals = x.shape[0] // 4
    x = jnp.reshape(x, (num_individuals, 4))

    leader_pos = x[0, 0:3]
    relative_positions = leader_pos - x[1:, 0:3]
    relative_ranges = jnp.linalg.norm(relative_positions, axis=1)
    return jnp.concatenate([leader_pos, relative_ranges])


MODELS["leader_follower_fo_kinematics"] = Model(
    dynamics=dynamics, observation=observation
)
