"""
Product-manifold retractions for states that mix Euclidean and SO(3) blocks.

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

A state vector ``x`` is a chart on a product manifold whose attitude factors are
unit quaternions. Treating that chart as flat -- adding rate vectors during a
rollout, or differentiating component-wise during Lie-derivative computation --
is only first-order correct and introduces a spurious radial gauge for every
quaternion. A :class:`Manifold` instead exposes the three operations needed to
work intrinsically:

* :meth:`Manifold.boxplus` -- the retraction ``x ⊞ xi`` mapping a tangent vector
  to a point on the manifold (Euclidean addition, ``q ⊗ Exp(xi)`` for attitude);
* :meth:`Manifold.to_tangent` -- the inverse map taking an ambient state rate to
  its tangent-space (body-rate) coordinates;
* :meth:`Manifold.plus_jacobian` -- the differential of ``boxplus`` at ``xi = 0``,
  the ``ambient_dim x tangent_dim`` map that reduces an ambient Jacobian to a
  covector on the tangent space.

``boxplus`` and ``plus_jacobian`` are a consistent retraction/linearization pair,
so an integrator and an observability Gramian built from the same manifold agree
to first order by construction.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla

from example_lib import math


class EuclideanBlock:
    r"""A flat :math:`\\mathbb{R}^n` factor; the retraction is plain addition."""

    def __init__(self, size: int):
        self.ambient_size = size
        self.tangent_size = size

    def boxplus(self, x, xi):
        return x + xi

    def to_tangent(self, x, dx):
        return dx

    def plus_jacobian(self, x):
        return jnp.eye(self.ambient_size)


class SO3Block:
    r"""A unit-quaternion (scalar-last) attitude factor on :math:`\\mathcal{S}^3`.

    Occupies four ambient components and three tangent (body-rate) DOF. The
    retraction is the right perturbation ``q ⊗ Exp(xi)``.
    """

    ambient_size = 4
    tangent_size = 3

    def boxplus(self, q, xi):
        return math.quaternion_product(q, math.quaternion_exp(xi))

    def to_tangent(self, q, dq):
        # Body rate omega from the ambient quaternion rate dq = J(q) omega, via
        # the pseudo-inverse J(q)^+ = (4 / |q|^2) J(q)^T (exact for the columns
        # q ⊗ [e_i/2, 0], which are orthogonal with squared norm |q|^2 / 4).
        jacobian = math.quaternion_plus_jacobian(q)
        return 4.0 / jnp.dot(q, q) * jacobian.T @ dq

    def plus_jacobian(self, q):
        return math.quaternion_plus_jacobian(q)


def Euclidean(size: int) -> "Manifold":
    """A purely Euclidean manifold of the given dimension (retraction = ``+``)."""
    return Manifold([EuclideanBlock(size)])


def SO3() -> SO3Block:
    """A standalone unit-quaternion attitude block."""
    return SO3Block()


class Manifold:
    """A product manifold assembled from ordered Euclidean/SO(3) blocks.

    Example -- the 16-component IMU nav-state ``[p, q, v, b_g, b_a]`` whose
    attitude is a quaternion at index 3::

        Manifold([EuclideanBlock(3), SO3Block(), EuclideanBlock(9)])

    has ``tangent_dim == 15``: the radial quaternion gauge is gone by construction.
    """

    def __init__(self, blocks):
        self.blocks = list(blocks)
        self._ambient_slices = []
        self._tangent_slices = []
        ambient = 0
        tangent = 0
        for block in self.blocks:
            self._ambient_slices.append(slice(ambient, ambient + block.ambient_size))
            self._tangent_slices.append(slice(tangent, tangent + block.tangent_size))
            ambient += block.ambient_size
            tangent += block.tangent_size
        self.ambient_dim = ambient
        self.tangent_dim = tangent

    @jax.named_scope("manifold.boxplus")
    def boxplus(self, x, xi):
        """Retract the tangent vector ``xi`` (size ``tangent_dim``) onto ``x``."""
        return jnp.concatenate([
            block.boxplus(x[a], xi[t])
            for block, a, t in zip(
                self.blocks, self._ambient_slices, self._tangent_slices, strict=True
            )
        ])

    @jax.named_scope("manifold.to_tangent")
    def to_tangent(self, x, dx):
        """Express the ambient state rate ``dx`` in tangent (body-rate) coordinates."""
        return jnp.concatenate([
            jnp.atleast_1d(block.to_tangent(x[a], dx[a]))
            for block, a in zip(self.blocks, self._ambient_slices, strict=True)
        ])

    @jax.named_scope("manifold.plus_jacobian")
    def plus_jacobian(self, x):
        """The ``ambient_dim x tangent_dim`` differential of ``boxplus`` at ``xi = 0``."""
        return jsla.block_diag(*[
            block.plus_jacobian(x[a])
            for block, a in zip(self.blocks, self._ambient_slices, strict=True)
        ])
