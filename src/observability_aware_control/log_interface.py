"""
Local Observability Gramian Interface.

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

import abc
from typing import Any

from jax import Array
from jax.typing import ArrayLike

from .typing import DynamicsFunction, ObservationFunction


class LocalObservabilityGramian(abc.ABC):
    """The interface for a Local Observability Gramian approximator.

    Essentially encodes that the local observability Gramian depends on the
    definition for both system dynamics and observation. Once initialized, LOG
    approximators can be called like any function to evaluate the LOG
    """

    @abc.abstractmethod
    def __init__(self, dynamics: DynamicsFunction, observation: ObservationFunction):
        """Initializes some Local Gramian approximation evaluator.

        Parameters
        ----------
        dynamics : DynamicsFunction
            A callable implementing the ODE for the dynamical system in the form
            f(x, u). Note no explicit time dependency is allowed
        observation : ObservationFunction
            A callable implementing the observation/output function for the
            dynamical system in the form h(x, u, *args). Feedforward is possible
            via `u` and arbitrary user data can be passed via *args
        """

    @abc.abstractmethod
    def __call__(self, x: ArrayLike, u: ArrayLike, dt: ArrayLike, *args: Any) -> Array:
        """Evaluates the Local Gramian approximation.

        Parameters
        ----------
        x : ArrayLike
            The operating state at which the Gramian is approximated
        u : ArrayLike
            Control inputs required to approximate the Gramian
        dt : ArrayLike
            Control timesteps or observation horizon (depending on approximation
            scheme definition)

        Returns
        -------
        jax.Array
            The approximated Local Observability Gramian (symmetric matrix with
            size equal to number of observations)
        """


LocalObservabilityGramianType = type[LocalObservabilityGramian]
