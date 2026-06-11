"""
Observability-aware cost function for optimal control.

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

from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
from jax.typing import ArrayLike

from . import integrator, log_interface, stlog
from .typing import DynamicsFunction, IndexExpression, ObservationFunction


class ObservabilityCostValue(NamedTuple):
    """A named tuple representing the return value of the observability cost."""

    objective: jax.Array
    gramians: ArrayLike | None = None
    states: ArrayLike | None = None
    inputs: ArrayLike | None = None


GramianMetric = Callable[[ArrayLike], jax.Array]


def _default_gramian_metric(gramians: ArrayLike):
    sigmas = jla.norm(gramians, axis=(1, 2), ord=-2)

    return 1e-3 / (sigmas.sum() + 1e-3)


class ObservabilityCost:
    """
    An observability-aware cost function.

    This cost function evaluates observability --- a metric of the local
    observability gramian, suitably approximated --- at each point along a
    model-prediction trajectory, then sums these observability metrics up to
    make for a scalar objective value.
    """

    def __init__(
        self,
        dynamics: DynamicsFunction,
        observation: ObservationFunction,
        integrator_dt: ArrayLike,
        *,
        gramian: log_interface.LocalObservabilityGramianType = stlog.STLOG,
        integration_method: integrator.Methods = integrator.Methods.RK4,
        gramian_kw: Mapping | None = None,
        gramian_metric: GramianMetric = _default_gramian_metric,
        observed_indices: IndexExpression = (),
    ):
        """Initializes the observability cost function object.

        Parameters
        ----------
        dynamics : typing.DynamicsFunction
            A callable implementing the ODE for the dynamical system in the form
            f(x, u). Note no explicit time dependency is allowed
        observation : typing.ObservationFunction
            A callable implementing the observation/output function for the
            dynamical system in the form h(x, u, *args). Feedforward is possible
            via `u` and arbitrary user data can be passed via *args
        integrator_dt : ArrayLike
            Stepsize for integrating the system dynamics during model prediction
        gramian : log_interface.LocalObservabilityGramianType, optional
            The type of a Local Observability Gramian evaluator satisfying the
            LocalObservabilityGramian interface, by default stlog.STLOG
        integration_method : integrator.Methods, optional
            A enumerator selecting the integration method for computing the
            system (and subsequently observation) trajectory, by default
            integrator.Methods.RK4
        gramian_kw : Optional[Mapping], optional
            A mapping of keywords to be passed into the initializer of the Local
            Observability Gramian, by default None
        gramian_metric : GramianMetric, optional
            A function that acts as a metric on Gramian metrics, by default
            default_gramian_metric
        observed_indices : typing.IndexExpression, optional
            An expression that indexes a subset of the system state whose
            observability is evaluated, by default ()
        """
        self._solve_ode = integrator.Integrator(
            dynamics, integration_method, stepsize=integrator_dt
        )
        self._gramian_metric = gramian_metric
        if gramian_kw is not None:
            self._stlog = gramian(dynamics, observation, **gramian_kw)
        else:
            self._stlog = gramian(dynamics, observation)

        if observed_indices:
            ix = (
                jnp.r_[observed_indices]
                if isinstance(observed_indices, slice)
                else jnp.asarray(observed_indices)
            )
            # Create slice that would extract rows and columns indexed by
            # 'observed_indices'
            self._stlog_slice = jnp.ix_(ix, ix)
        else:
            # Take everything
            self._stlog_slice = ...

    def eval_integrator(self, x0, us):
        """Evaluates the integrator to produce a state trajectory."""
        return self._solve_ode(x0, us)

    def eval_gramian(
        self, x: ArrayLike, u: ArrayLike, dt: ArrayLike, *args: Any
    ) -> jax.Array:
        """Invokes the underlying STLOG approximator.

        The slicing scheme specified by 'observed_indices' in the initializer slices
        the Gramian appropriately before returning

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
            The Local Observability Gramian approximated by the current scheme
            and sliced appropriately
        """
        return self._stlog(x, u, dt, *args)[self._stlog_slice]

    def __call__(
        self,
        x0: ArrayLike,
        us: ArrayLike,
        dt: ArrayLike,
        gramian_args: Any = (),
        return_trajectory: bool = False,
        return_gramians: bool = False,
    ) -> ObservabilityCostValue:
        """Evaluates the observability cost.

        Given an initial state and a sequence of control inputs, this function
        evaluating a metric of the observability gramian at each point of the
        resultant state trajectory and summing up the results

        Parameters
        ----------
        x0 : ArrayLike
            The initial state
        us : ArrayLike
            A sequence (array) of control inputs. The length of this sequence
            determines the number of prediction steps to be taken
        dt : ArrayLike
            Observation horizon(s). If this is a scalar, then the same
            observation horizon will be used at every prediction step
        gramian_args : Any, optional
            Any additional arguments to be passed to the Gramian approximator
            (the underlying observation function), by default ()
        return_trajectory : bool, optional
            Toggles if the predicted trajectory (shooting nodes) will be
            returned, by default False
        return_gramians : bool, optional
            Toggles if all the Gramians at each point on the predicted
            trajectory will be returned, by default False

        Returns
        -------
        ObservabilityCostValue
            A tuple consisting of (cost_value [, gramians [, states, inputs]]),
            where gramians, states, and inputs may be None if return_{gramians,
            trajectory} are not toggled on
        """
        us = jnp.asarray(us)
        xs, _ = self._solve_ode(x0, us)

        dt = jnp.asarray(dt)

        dt = jnp.broadcast_to(dt, us.shape[0])
        gramians = jax.vmap(self.eval_gramian)(xs, us, dt, *gramian_args)

        objective = self._gramian_metric(gramians)
        if return_trajectory and return_gramians:
            return ObservabilityCostValue(objective, gramians, xs, us)

        if return_trajectory:
            return ObservabilityCostValue(objective, states=xs, inputs=us)

        if return_gramians:
            return ObservabilityCostValue(objective, gramians)

        return ObservabilityCostValue(objective)
