"""
Observability Aware Controller.

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

import functools as fn
import inspect
from typing import Any

import equinox as eqx
import jax
from jax.experimental import checkify
import jax.numpy as jnp
from jax.typing import ArrayLike
from scipy import optimize

from observability_aware_control import utils

from . import observability_cost
from .typing import ConstraintFunction, IndexExpression


def _recombine_input(u, u_const, id_mut, id_const):
    u = u.reshape(len(u_const), -1)
    return utils.combine_array(u, u_const, id_mut, id_const)


class ObservabilityAwareController:
    """
    An observability-aware optimal controller.

    This controller solves our Observability-Aware control problem, computing
    a sequence of control inputs that minimize the sum of the minimum singular
    value of the Local Observability Gramian at each point along a predicted
    trajectory
    """

    def __init__(
        self,
        cost: observability_cost.ObservabilityCost,
        lb: ArrayLike = -jnp.inf,
        ub: ArrayLike = jnp.inf,
        method: utils.optim_utils.Method | None = None,
        optim_options: dict[str, Any] | None = None,
        constraint: ConstraintFunction | None = None,
        constraint_bounds: tuple[ArrayLike, ArrayLike] = (-jnp.inf, 0.0),
    ):
        """Initializes the Observability Aware Controller.

        Parameters
        ----------
        cost : observability_cost.ObservabilityCost
            An already-initialized ObservabilityCost object
        lb : ArrayLike, optional
            Control input lower bounds. Can be a 1D array which is uniform
            bounds over all predictive timesteps, or a matrix containing bounds
            stacked over first axis, by default -jnp.inf
        ub : ArrayLike, optional
            Control input lower bounds. Can be a 1D array which is uniform
            bounds over all predictive timesteps, or a matrix containing bounds
            stacked over first axis, by default jnp.inf
        method : Optional[utils.optim_utils.Method], optional
            Algorithm to be used by scipy.optimize.minimize, by default None
        optim_options : Optional[Dict[str, Any]], optional
            Options to be passed to scipy.optimize.minimize, by default None
        constraint : Optional[ConstraintFunction], optional
            Nonlinear constraint function, by default None
        constraint_bounds : Tuple[ArrayLike, ArrayLike], optional
            Bounds for nonlinear constraints, by default (-jnp.inf, 0.0)
        """
        self._cost_fcn = cost

        self._grad_fcn = jax.grad(lambda us, x, dt: self._cost_fcn(x, us, dt).objective)

        self._problem = utils.optim_utils.MinimizeProblem(
            self._objective,
            jac=self._gradient,
            method=method,
            options=optim_options,
        )

        self._lb = jnp.asarray(lb)
        self._ub = jnp.asarray(ub)

        if constraint is not None and constraint_bounds is not None:
            self._constr_fcn = constraint
            self._constraint_lb, self._constraint_ub = jnp.broadcast_arrays(
                *map(jnp.asarray, constraint_bounds)
            )
            self._constraint_jac_fcn = jax.jacobian(
                lambda us, x, *a, **kw: constraint(x, us, *a, **kw)
            )
        else:
            self._constr_fcn = None

    @eqx.filter_jit
    def _objective(self, u, u_const, x, dt, minimized_indices, constant_indices):
        u = _recombine_input(u, u_const, minimized_indices, constant_indices)
        return self._cost_fcn(x, u, dt).objective

    @eqx.filter_jit
    def _gradient(self, u, u_const, x, dt, minimized_indices, constant_indices):
        u = _recombine_input(u, u_const, minimized_indices, constant_indices)
        return self._grad_fcn(u, x, dt)[..., minimized_indices].ravel()

    @eqx.filter_jit
    def _constraint(self, u, u_const, x, _, minimized_indices, constant_indices):
        assert self._constr_fcn  # noqa

        u = _recombine_input(u, u_const, minimized_indices, constant_indices)

        # If the constraint model requires the cost model, then pass it in
        if "cost" in inspect.signature(self._constr_fcn).parameters:
            return self._constr_fcn(x, u, cost=self._cost_fcn)
        return self._constr_fcn(x, u)

    @eqx.filter_jit
    def _constraint_jacobian(
        self, u, u_const, x, _, minimized_indices, constant_indices
    ):
        u = _recombine_input(u, u_const, minimized_indices, constant_indices)
        jac = self._constraint_jac_fcn(u, x, cost=self._cost_fcn)
        return jac[..., minimized_indices].reshape(jac.shape[0], -1)

    def minimize(
        self,
        x0: ArrayLike,
        u0: ArrayLike,
        t: ArrayLike,
        minimized_indices: IndexExpression = (),
    ) -> optimize.OptimizeResult:
        """Solves the Observability-Aware Control Problem.

        Parameters
        ----------
        x0 : ArrayLike
            Initial state for model prediction
        u0 : ArrayLike
            Initial guess of the control inputs for model prediction
        t : ArrayLike
            The observation horizon
        minimized_indices : IndexExpression, optional
            An expression that indexes the components in the control input to be
            minimized, by default ()

        Returns
        -------
        optimize.OptimizeResult
            The result of invoking scipy.optimize.minimize
        """
        u0 = jnp.asarray(u0)
        num_steps, num_inputs = u0.shape
        if not minimized_indices:
            minimized_idx = jnp.arange(num_inputs)
            constant_idx = ()
        else:
            minimized_idx = jnp.asarray(minimized_indices)
            constant_idx = jnp.setdiff1d(jnp.arange(num_inputs), minimized_idx)

        *bounds, u0 = jnp.broadcast_arrays(self._lb, self._ub, u0)
        # Broadcast bounds over all time windows, takes mutable components, then flatten
        self._problem.bounds = optimize.Bounds(
            *(it[..., minimized_idx].ravel() for it in bounds)  # type: ignore
        )

        u_const = u0[..., constant_idx]
        self._problem.x0 = u0[..., minimized_idx].ravel()
        args = (u_const, x0, t, minimized_idx, constant_idx)
        self._problem.args = args

        if self._constr_fcn is not None:
            if self._constraint_lb.ndim <= 1:
                self._constraint_lb = jnp.atleast_1d(self._constraint_lb)

                bcast = fn.partial(
                    jnp.broadcast_to, shape=(num_steps, len(self._constraint_lb))
                )
                lb, ub = map(bcast, (self._constraint_lb, self._constraint_ub))
            else:
                lb, ub = self._constraint_lb, self._constraint_ub
            self._problem.constraints = optimize.NonlinearConstraint(
                lambda u: self._constraint(u, *args),
                *map(jnp.ravel, (lb, ub)),
                jac=lambda u: self._constraint_jacobian(u, *args),  # type: ignore
            )

        rec = utils.optim_utils.OptimizationRecorder()
        self._problem.callback = rec.update

        prob_dict = vars(self._problem)
        soln = optimize.minimize(**prob_dict)

        soln["x"] = _recombine_input(soln.x, u_const, minimized_idx, constant_idx)
        soln["fun_hist"] = rec.fun
        return soln
