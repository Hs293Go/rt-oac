"""
Unified point-feature inertial-odometry model (visual / LIDAR).

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

The exteroceptive measurement of a 3-D feature is the only modality-specific
piece; the dynamics is the universal IMU strapdown nav-state, control-affine in
the IMU measurements. Selecting the observation `kind` instantiates a monocular
pinhole, a LIDAR point-to-plane, or a range (UWB-like) sensor on the same model.
A static feature (passed as data) is ego-motion odometry; promoting the feature
to a state with its own dynamics recovers relative localization.
"""

import enum
import functools

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla

from example_lib import manifold as manifold_lib
from example_lib import math
from example_lib.models import MODELS, Model

NUM_STATES = 16  # [p(3), q(4), v(3), b_g(3), b_a(3)]
NUM_INPUTS = 6  # [omega_m(3), accel_m(3)] -- gyroscope, accelerometer

GRAVITY = jnp.array([0.0, 0.0, -9.81])

# The 15-DOF tangent structure of the nav-state: the attitude is the unit
# quaternion at index 3, every other block is Euclidean. Supplying this to an
# integrator advances attitude by q ⊗ Exp(omega dt); supplying it to the STLOG
# yields a 15x15 tangent-space Gramian with no redundant quaternion gauge.
MANIFOLD = manifold_lib.Manifold([
    manifold_lib.EuclideanBlock(3),  # p
    manifold_lib.SO3Block(),  # q
    manifold_lib.EuclideanBlock(9),  # v, b_g, b_a
])


@jax.jit
def dynamics(x: jax.Array, u: jax.Array) -> jax.Array:
    """IMU strapdown kinematics, control-affine in the IMU measurements.

    State ``x = [p, q, v, b_g, b_a]`` holds the world-frame position and
    velocity, the body-to-world quaternion (scalar-last), and the gyroscope and
    accelerometer biases. Input ``u = [omega_m, accel_m]``. The quaternion is
    normalized so the model depends only on the *unit* attitude. Build Gramians
    against :data:`MANIFOLD` so the attitude is differentiated on the tangent
    space of :math:`\\mathcal{S}^3`; the redundant radial quaternion direction is
    then absent by construction rather than appearing as a null direction to be
    projected out afterwards.
    """
    p = x[0:3]
    q = x[3:7] / jla.norm(x[3:7])
    v = x[7:10]
    b_g = x[10:13]
    b_a = x[13:16]

    omega = u[0:3] - b_g
    accel = u[3:6] - b_a

    half_omega = jnp.array([omega[0] / 2.0, omega[1] / 2.0, omega[2] / 2.0, 0.0])
    return jnp.concatenate([
        v,
        math.quaternion_product(q, half_omega),
        math.quaternion_rotate_point(q, accel) + GRAVITY,
        jnp.zeros(3),
        jnp.zeros(3),
    ])


class ObservationKind(enum.Enum):
    PINHOLE = 0  # monocular bearing to known world points (2 scalars / feature)
    POINT_TO_PLANE = 1  # LIDAR plane distance of a frozen body point (1 / feature)
    RANGE = 2  # range to known world points (1 / feature)


def observation(x, _, features, *, kind=ObservationKind.PINHOLE):
    """Stacked exteroceptive measurement of a set of features.

    ``features`` is a ``(K, 3)`` array of world points for ``PINHOLE`` and
    ``RANGE``; for ``POINT_TO_PLANE`` it is a ``(K, 7)`` array of rows
    ``[ell_b(3), n_w(3), d_w(1)]``, where ``ell_b`` is the body-frame coordinate
    of a frozen correspondence to the world plane ``(n_w, d_w)`` (held constant
    over the short observation horizon). The second argument (the input ``u``)
    is unused.
    """
    p = x[0:3]
    q = x[3:7] / jla.norm(x[3:7])

    if kind == ObservationKind.PINHOLE:

        def one(p_w):
            c = math.quaternion_rotate_point(q, p_w - p, invert_rotation=True)
            return jnp.array([c[0] / c[2], c[1] / c[2]])

        return jax.vmap(one)(features).ravel()
    if kind == ObservationKind.RANGE:
        return jax.vmap(lambda p_w: jla.norm(p_w - p))(features)
    if kind == ObservationKind.POINT_TO_PLANE:

        def one(f):
            ell_b, n_w, d_w = f[0:3], f[3:6], f[6]
            return n_w @ (math.quaternion_rotate_point(q, ell_b) + p) + d_w

        return jax.vmap(one)(features)
    raise ValueError(f"{kind} is not a valid kind of point-feature observation")


def min_observable_eigenvalue(gramian: jax.Array, x: jax.Array = None) -> jax.Array:
    """Worst-direction observability of the 15-DOF nav-state.

    With a tangent-space (15x15) STLOG -- built against :data:`MANIFOLD` -- the
    quaternion gauge is already absent, so the worst observable direction is
    simply the smallest eigenvalue. It collapses toward 0 under degeneracy. The
    ``x`` argument is accepted for backward compatibility and ignored.
    """
    return jla.eigvalsh(gramian)[0]


MODELS["pinhole_inertial"] = Model(
    dynamics=dynamics,
    observation=functools.partial(observation, kind=ObservationKind.PINHOLE),
    manifold=MANIFOLD,
)
MODELS["point_to_plane_inertial"] = Model(
    dynamics=dynamics,
    observation=functools.partial(observation, kind=ObservationKind.POINT_TO_PLANE),
    manifold=MANIFOLD,
)
MODELS["range_inertial"] = Model(
    dynamics=dynamics,
    observation=functools.partial(observation, kind=ObservationKind.RANGE),
    manifold=MANIFOLD,
)
