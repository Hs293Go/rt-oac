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

import equinox as eqx
import jax
import jax.numpy as jnp

from example_lib import manifold as manifold_lib
from example_lib import math
from example_lib.models import MODELS, Model

NUM_STATES = 10
NUM_INPUTS = 8

# State [r_lf(3), q_fl(4), v_lf(3)]: a relative position, the relative attitude
# quaternion, and a relative velocity -- 9 tangent DOF.
MANIFOLD = manifold_lib.Manifold([
    manifold_lib.EuclideanBlock(3),
    manifold_lib.SO3Block(),
    manifold_lib.EuclideanBlock(3),
])


@jax.jit
def dynamics(x: jax.Array, u: jax.Array):
    r_lf, q_fl, v_lf = x[:3], x[3:7], x[7:10]

    f_l, omega_l, f_f, omega_f = u[0], u[1:4], u[4], u[5:8]

    w_f = jnp.array([omega_f[0] / 2.0, omega_f[1] / 2.0, omega_f[2] / 2.0, 0.0])
    w_l = jnp.array([omega_l[0] / 2.0, omega_l[1] / 2.0, omega_l[2] / 2.0, 0.0])

    t_l, t_f = jnp.array([0.0, 0.0, f_l]), jnp.array([0.0, 0.0, f_f])
    return jnp.concatenate([
        jnp.cross(r_lf, omega_f) + v_lf,
        -math.quaternion_product(w_f, q_fl) + math.quaternion_product(q_fl, w_l),
        jnp.cross(v_lf, omega_f) + math.quaternion_rotate_point(q_fl, t_l) - t_f,
    ])


@eqx.filter_jit
def observation(x: jax.Array, _: jax.Array | None = None, *, use_sqnorm=True):
    r_lf = x[0:3]
    q_fl = x[3:7]

    r_lf_sqnorm = jnp.dot(r_lf, r_lf)
    range_meas = r_lf_sqnorm if use_sqnorm else jnp.sqrt(r_lf_sqnorm)
    return jnp.concatenate([jnp.array([range_meas]), q_fl])


def interrobot_distance_squared(x0, us, cost):
    xs, _ = cost.eval_integrator(x0, us)
    return jnp.sum(xs[:, 0:3] ** 2, axis=1)


@jax.jit
def to_absolute_state(x_l, x_lf):
    """Convert relative state to absolute state."""
    p_l = x_l[:3]
    q_l = x_l[3:7]
    v_l = x_l[7:10]

    p_lf = x_lf[:3]
    q_fl = x_lf[3:7]
    v_lf = x_lf[7:10]

    q_f = math.quaternion_product(q_l, math.quaternion_inverse(q_fl))
    p_f = p_l - math.quaternion_rotate_point(q_f, p_lf)
    v_f = v_l - math.quaternion_rotate_point(q_f, v_lf)
    return jnp.concatenate([p_l, q_l, v_l, p_f, q_f, v_f])


@jax.jit
def from_absolute_state(x_l, x_f):
    """Convert absolute state to relative state."""
    p_l = x_l[:3]
    q_l = x_l[3:7]
    v_l = x_l[7:10]

    p_f = x_f[:3]
    q_f = x_f[3:7]
    v_f = x_f[7:10]

    q_fi = math.quaternion_inverse(q_f)
    q_fl = math.quaternion_product(q_fi, q_l)
    p_lf = math.quaternion_rotate_point(q_fi, p_l - p_f)
    v_lf = math.quaternion_rotate_point(q_fi, v_l - v_f)
    return jnp.concatenate([p_lf, q_fl, v_lf])


MODELS["inter_quadrotor_relative_dynamics"] = Model(
    dynamics=dynamics, observation=observation, manifold=MANIFOLD
)
