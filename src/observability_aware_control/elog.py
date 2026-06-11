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
from jax.typing import ArrayLike

from . import integrator, log_interface
from .typing import DynamicsFunction, IndexExpression, ObservationFunction


class ELOG(log_interface.LocalObservabilityGramian):
    """Krener and IDE's Empirical Local Observability Gramian (ELOG) relies on
    numerically differentiating a path of observations. This is accomplished by.

    * perturbing each component of the initial condition
    * For each perturbed initial condition, compute the resultant system
      trajectory by numerical integration
    * Compute a path of observations corresponding to each trajectory
    * Take the difference between pairs of observation paths
    * Take the the inner product between observation differences
    """

    def __init__(
        self,
        dynamics: DynamicsFunction,
        observation: ObservationFunction,
        eps: ArrayLike = 1e-3,
        perturb_indices: IndexExpression = ...,
        method: integrator.Methods = integrator.Methods.RK4,
    ):
        """Initializes the ELOG Local Gramian approximation evaluator.

        Parameters
        ----------
        dynamics : DynamicsFunction
            A callable implementing the ODE for the dynamical system in the form
            f(x, u). Note no explicit time dependency is allowed
        observation : ObservationFunction
            A callable implementing the observation/output function for the
            dynamical system in the form h(x, u, *args). Feedforward is possible
            via `u` and arbitrary user data can be passed via *args
        eps : ArrayLike, optional
            The perturbation applied to the initial condition. May be a scalar,
            in which case the same perturbation is applied to all components
            uniformly, or an array of perturbations matched to each state
            component to be perturbed, by default 1e-3
        perturb_indices : IndexExpression, optional
            An expression that indexes a subset of the system state to be
            perturbed, by default ()
        method : integrator.Methods, optional
            A enumerator selecting the integration method for computing the
            system (and subsequently observation) trajectory, by default
            integrator.Methods.RK4
        """
        self._solve_ode = integrator.Integrator(dynamics, method)
        self._observation = jax.vmap(observation)
        self._eps = jnp.asarray(eps)
        self._perturb_indices = perturb_indices
        self._method = method

    def __call__(
        self, x0: ArrayLike, u: ArrayLike, dt: ArrayLike, *args: Any
    ) -> jax.Array:
        """Evaluates the ELOG Local Gramian approximate starting from a given
        initial condition through a trajectory of control inputs.

        Parameters
        ----------
        x : ArrayLike
            The operating state at which the Gramian is approximated
        u : ArrayLike
            An array (sequence) of control inputs spanning the full trajectory
            over which observability is evaluated
        dt : ArrayLike
            An array of timesteps between application of control inputs,
            spanning the full trajectory. Can be a scalar, in which case a
            single uniform timestep is used.

            Warning: The sum of all timesteps is the full horizon

        Returns
        -------
        jax.Array
            The Local Observability Gramian (symmetric matrix with size equal to
            number of observations) approximated by the ELOG scheme
        """
        x0 = jnp.asarray(x0)
        u = jnp.asarray(u)
        dt /= len(u)
        self._solve_ode.stepsize = dt

        def _perturb(x0_plus, x0_minus):
            xs_plus, _ = self._solve_ode(x0_plus, u)
            yi_plus = self._observation(xs_plus, u, *args)
            xs_minus, _ = self._solve_ode(x0_minus, u)
            yi_minus = self._observation(xs_minus, u, *args)
            return jnp.atleast_2d(yi_plus - yi_minus)

        perturb_bases = jnp.eye(len(x0))[self._perturb_indices, :]
        x0_plus = x0 + self._eps * perturb_bases
        x0_minus = x0 - self._eps * perturb_bases
        y_all = jax.vmap(_perturb, out_axes=-1)(x0_plus, x0_minus) / (2.0 * self._eps)
        dt = jnp.broadcast_to(dt, (len(u),) + (1,) * 3)

        return jnp.sum(dt * (y_all[..., None] * y_all[..., None].mT), axis=(0, 1))
