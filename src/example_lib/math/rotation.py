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

import equinox as eqx
import jax
import jax.numpy as jnp


@eqx.filter_jit
def angle_rotate_point(angle, point, invert_rotation=False):
    if invert_rotation:
        angle = -angle
    c = jnp.cos(angle)
    s = jnp.sin(angle)

    return jnp.array([
        c * point[0] - s * point[1],
        s * point[0] + c * point[1],
    ])


@jax.jit
def quaternion_product(lhs, rhs):
    return jnp.array([
        lhs[3] * rhs[0] + lhs[0] * rhs[3] + lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[3] * rhs[1] + lhs[1] * rhs[3] + lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[3] * rhs[2] + lhs[2] * rhs[3] + lhs[0] * rhs[1] - lhs[1] * rhs[0],
        lhs[3] * rhs[3] - lhs[0] * rhs[0] - lhs[1] * rhs[1] - lhs[2] * rhs[2],
    ])


@eqx.filter_jit
def quaternion_rotate_point(quaternion, point, invert_rotation=False):
    vec = -quaternion[0:3] if invert_rotation else quaternion[0:3]
    uv = jnp.cross(vec, point)
    uv += uv
    return point + quaternion[3] * uv + jnp.cross(vec, uv)


@jax.jit
def quaternion_inverse(quaternion):
    return jnp.array([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]])


# A rotation vector with squared norm below this is treated with the small-angle
# Taylor expansions of the exponential/logarithm to avoid the 0/0 in
# ``sin(theta/2) / theta``; the cut-off is well inside float64 accuracy. The
# branch that divides is additionally fed a *clamped* squared norm so that
# ``sqrt`` has a finite gradient at zero (the unused branch is still
# differentiated under reverse-mode AD).
_SMALL_ANGLE_SQ = 1e-12


@jax.jit
def quaternion_exp(rotvec):
    r"""The unit quaternion :math:`\mathrm{Exp}(\boldsymbol\xi)` of a rotation vector.

    This is the Lie-group exponential :math:`\mathfrak{so}(3) \to \mathcal{S}^3`
    in the scalar-last convention, ``Exp(xi) = [sin(|xi|/2) xi/|xi|, cos(|xi|/2)]``.
    Composed on the right of an attitude, ``q ⊗ Exp(xi)`` rotates by ``xi``
    expressed in the *body* frame -- the retraction used to integrate attitude
    and to define tangent-space derivatives (see :func:`quaternion_plus_jacobian`).
    """
    theta_sq = rotvec @ rotvec
    small = theta_sq < _SMALL_ANGLE_SQ
    # Clamp the radicand so ``sqrt`` (and the ``/ theta`` below) have finite
    # gradients even at the origin; the clamped value is masked out by ``where``.
    theta = jnp.sqrt(jnp.where(small, 1.0, theta_sq))
    # ``sin(theta/2) / theta`` -> ``1/2 - theta^2/48`` as theta -> 0.
    vec_coeff = jnp.where(small, 0.5 - theta_sq / 48.0, jnp.sin(theta / 2.0) / theta)
    scalar = jnp.where(small, 1.0 - theta_sq / 8.0, jnp.cos(theta / 2.0))
    return jnp.append(vec_coeff * rotvec, scalar)


@jax.jit
def quaternion_log(quaternion):
    r"""The rotation vector :math:`\mathrm{Log}(\mathbf q)` of a unit quaternion.

    Inverse of :func:`quaternion_exp` over ``|xi| < 2*pi``, returning the body-frame
    rotation vector such that ``q ⊗ Exp(-Log(q)) = identity``.
    """
    vec = quaternion[0:3]
    scalar = quaternion[3]
    vec_norm_sq = vec @ vec
    small = vec_norm_sq < _SMALL_ANGLE_SQ
    vec_norm = jnp.sqrt(jnp.where(small, 1.0, vec_norm_sq))
    # angle = 2 * atan2(|vec|, scalar); axis = vec / |vec|.
    angle = 2.0 * jnp.arctan2(vec_norm, scalar)
    # ``angle / |vec|`` -> ``2 / scalar`` as |vec| -> 0 (small-angle limit).
    coeff = jnp.where(small, 2.0 / scalar, angle / vec_norm)
    return coeff * vec


@jax.jit
def quaternion_plus_jacobian(quaternion):
    r"""The :math:`4 \times 3` differential of the right retraction ``q ⊗ Exp(xi)``.

    Returns ``d(q ⊗ Exp(xi))/d xi`` at ``xi = 0``, i.e. the map from a body-frame
    angular perturbation to the induced quaternion increment. Its ``i``-th column
    is ``q ⊗ [e_i/2, 0]``; equivalently ``q_dot = J(q) omega`` is exactly the
    attitude kinematics ``q ⊗ [omega/2, 0]``. This is the tangent map that turns
    an ambient (4-component) quaternion Jacobian into a proper 3-DOF covector on
    :math:`T_q \mathcal{S}^3`.
    """
    x, y, z, w = quaternion
    return 0.5 * jnp.array([
        [w, -z, y],
        [z, w, -x],
        [-y, x, w],
        [-x, -y, -z],
    ])
