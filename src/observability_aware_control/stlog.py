"""
The short time local observability gramian (STLOG) implementation.

Copyright © 2024 H S Helson Go and Ching Lok Chong

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

from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy import special
from jax.typing import ArrayLike

from . import lie_derivative, log_interface
from .typing import DynamicsFunction, ObservationFunction


class STLOG(log_interface.LocalObservabilityGramian):
    """

    The Short Time Local Observability Gramian.

    It is a novel approximation of the
    Local Observability Gramian that represents the ability to uniquely
    determine the current system state from prospective observations along a
    trajectory a "short time" into the future
    """

    def __init__(
        self,
        dynamics: DynamicsFunction,
        observation: ObservationFunction,
        order: int = 1,
        var: ArrayLike | None = None,
        manifold=None,
    ):
        """Initializes the STLOG Local Gramian approximation evaluator.

        Parameters
        ----------
        dynamics : DynamicsFunction
            A callable implementing the ODE for the dynamical system in the form
            f(x, u). Note no explicit time dependency is allowed
        observation : ObservationFunction
            A callable implementing the observation/output function for the
            dynamical system in the form h(x, u, *args). Feedforward is possible
            via `u` and arbitrary user data can be passed via *args
        order : int, optional
            The order of approximation, by default 1
        var : Optional[ArrayLike], optional
            Observation variance encoding noise/uncertainty affecting each
            observation, by default None
        manifold : Manifold, optional
            A product-manifold retraction (see ``example_lib.manifold``). When
            given, each Lie-derivative gradient ``D(L_f^j h)`` is reduced from an
            ambient row covector to a covector on the tangent space ``T_x M`` by
            right-multiplying with the manifold's ``plus_jacobian``, so the STLOG
            is the genuine ``tangent_dim x tangent_dim`` Gramian -- the redundant
            quaternion gauge is removed analytically rather than projected out
            afterwards. When ``None`` (the default), the ambient component-wise
            Gramian is returned, reproducing the previous behaviour.
        """
        self._manifold = manifold
        self._lie_derivative_gradients = [
            jax.jacobian(it)
            for it in lie_derivative.lie_derivative(observation, dynamics, order)
        ]
        if var is None:
            self._inv_cov = None
        else:
            self._inv_cov = jnp.asarray(jnp.diag(1.0 / var)[jnp.newaxis, jnp.newaxis])

        self._order = order

        order_seq = jnp.arange(order + 1)
        self._a, self._b, *_ = jnp.ix_(order_seq, order_seq)
        self._k = self._a + self._b + 1
        facts = special.factorial(order_seq)
        self._den = facts[self._a] * facts[self._b] * self._k

    @property
    def order(self):
        """The order of approximation of the STLOG."""
        return self._order

    def __call__(
        self, x: ArrayLike, u: ArrayLike, dt: ArrayLike, *args: Any
    ) -> jax.Array:
        """Evaluates the STLOG at a given state and control input.

        Parameters
        ----------
        x : ArrayLike
            The operating state at which the Gramian is approximated
        u : ArrayLike
            The control input at which the Gramian is approximated
        dt : ArrayLike
            The observation horizon
        *args : Any
            Arbitrary user data passed to the observation function

        Returns
        -------
        jax.Array
            The Local Observability Gramian (symmetric matrix with size equal to
            number of observations) approximated by the STLOG scheme
        """
        if self._manifold is None:
            lie_derivative_gradients = jnp.stack([
                jnp.atleast_2d(it(x, u, *args)) for it in self._lie_derivative_gradients
            ])
        else:
            # Reduce each ambient gradient D(L_f^j h) to a covector on T_x M.
            tangent_map = self._manifold.plus_jacobian(x)
            lie_derivative_gradients = jnp.stack([
                jnp.atleast_2d(it(x, u, *args)) @ tangent_map
                for it in self._lie_derivative_gradients
            ])

        factor = jnp.asarray(dt) ** self._k / self._den

        if self._inv_cov is None:
            return jnp.sum(
                factor[..., None, None]
                * lie_derivative_gradients[self._a, ...].mT
                @ lie_derivative_gradients[self._b, ...],
                axis=(0, 1),
            )

        return jnp.sum(
            factor[..., None, None]
            * lie_derivative_gradients[self._a].mT
            @ self._inv_cov
            @ lie_derivative_gradients[self._b],
            axis=(0, 1),
        )
