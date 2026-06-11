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

import enum

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla

from example_lib import math


@jax.custom_jvp
@jax.jit
def dynamics(x, u):
    heading = x[2]
    velocity, angular_velocity = map(jnp.squeeze, jnp.split(u, [1]))
    return jnp.append(
        math.angle_rotate_point(heading, jnp.array([velocity, 0.0])),
        angular_velocity,
    )


@dynamics.defjvp
@jax.jit
def dynamics_jvp(primals, tangents):
    x, u = primals
    dx, du = tangents

    velocity, yaw_rate = u
    heading = x[2]

    sx, cx = jnp.sin(heading), jnp.cos(heading)

    fval = jnp.array([cx * velocity, sx * velocity, yaw_rate])
    jvp_val = jnp.array([
        -sx * velocity * dx[2] + cx * du[0],
        cx * velocity * dx[2] + sx * du[0],
        du[1],
    ])

    return fval, jvp_val


class ObservationKind(enum.Enum):
    POSITION = 0
    RANGE = 1
    # BEARING = 2


def observation(x, _, lm, kind=ObservationKind.POSITION):
    position, heading, *_ = map(jnp.squeeze, jnp.split(x, [2]))
    relative_position = math.angle_rotate_point(heading, lm - position, True)
    if kind == ObservationKind.POSITION:
        return jnp.append(relative_position, heading)
    if kind == ObservationKind.RANGE:
        return jnp.array([jla.norm(relative_position), heading])
    # if kind == ObservationKind.BEARING:
    #     return jnp.array([
    #         jnp.arctan2(relative_position[1], relative_position[0]),
    #         heading,
    #     ])
    raise ValueError(f"{kind} is not a valid kind of interrobot observation")
