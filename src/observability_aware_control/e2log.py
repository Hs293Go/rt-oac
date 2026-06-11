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

from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy import special
from jax.typing import ArrayLike

from observability_aware_control import integrator

from . import lie_derivative, log_interface
from .typing import DynamicsFunction, ObservationFunction


class E2LOG(log_interface.LocalObservabilityGramian):
    """Hausmann and Priess's Empirical Expanded Local Observability Gramian."""

    def __init__(
        self,
        dynamics: DynamicsFunction,
        observation: ObservationFunction,
        order: int = 1,
        method: integrator.Methods = integrator.Methods.RK4,
        manifold=None,
    ):
        self._manifold = manifold
        self._solve_ode = integrator.Integrator(dynamics, method, manifold=manifold)
        self._lie_derivative_gradients = [
            jax.jacobian(it)
            for it in lie_derivative.lie_derivative(observation, dynamics, order)
        ]
        self._order_seq = jnp.arange(order + 1)
        self._facts = special.factorial(self._order_seq)

    def __call__(self, x0: ArrayLike, us: ArrayLike, dt: ArrayLike, *args: Any):
        x0 = jnp.asarray(x0)
        us = jnp.asarray(us)
        dt /= len(us)
        self._solve_ode.stepsize = dt

        k = (dt**self._order_seq / self._facts)[..., None, None]

        @jax.vmap
        def _eval_lie_derivative_product(x, u, *args):
            if self._manifold is None:
                gradients = [
                    jnp.atleast_2d(it(x, u, *args))
                    for it in self._lie_derivative_gradients
                ]
            else:
                tangent_map = self._manifold.plus_jacobian(x)
                gradients = [
                    jnp.atleast_2d(it(x, u, *args)) @ tangent_map
                    for it in self._lie_derivative_gradients
                ]
            lie_derivative_gradients = jnp.stack(gradients) * k

            return (lie_derivative_gradients.mT @ lie_derivative_gradients).sum(axis=0)

        xs, _ = self._solve_ode(x0, us)

        return dt * _eval_lie_derivative_product(xs, us, *args).sum(axis=0)
