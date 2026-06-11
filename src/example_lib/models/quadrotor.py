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

import jax.numpy as jnp

from example_lib import math

NUM_STATES = 10


class InputKind(enum.Enum):
    THRUST = 0
    ACCELERATION = 1


def dynamics(x, u, input_kind=InputKind.THRUST):
    _, q, v = jnp.split(x, [3, 7])

    if input_kind == InputKind.THRUST:
        thrust, body_rate = jnp.split(u, [1])
        accel = jnp.array([0.0, 0.0, thrust.squeeze()])
    elif input_kind == InputKind.ACCELERATION:
        accel, body_rate = jnp.split(u, [3])
    w = jnp.append(body_rate / 2.0, 0.0)
    g = jnp.array([0.0, 0.0, -9.81])

    dx = jnp.empty(NUM_STATES)
    dx = dx.at[0:3].set(v)
    dx = dx.at[3:7].set(math.quaternion_product(q, w))
    dx = dx.at[7:10].set(math.quaternion_rotate_point(q, accel) + g)
    return dx
