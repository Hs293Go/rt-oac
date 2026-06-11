"""
Integrator for continuous-time dynamical systems.

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

import enum
import functools

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .typing import DynamicsFunction, OutputFunction


class Methods(enum.Enum):
    """An enumerator of integration methods."""

    EULER = 0
    RK4 = 1


class Integrator:
    """
    An ODE integrator for continuous-time dynamical systems.

    Our Integrator solves an ODE representing an continuous-time dynamical
    system by integration. Our ODEs must be autonomous and implicitly depend on
    time through time-dependent control inputs. Such control inputs must be
    discretized a-priori and be piecewise constant, aka they are specified as a
    sequence of arrays corresponding to given time points. Our integrator steps
    through these time points and do not carry out adaptive sizing.
    """

    def __init__(
        self,
        dynamics: DynamicsFunction,
        method: Methods = Methods.RK4,
        output: OutputFunction | None = None,
        stepsize: ArrayLike = 1.0,
        manifold=None,
    ):
        """Initializes the Integrator object.

        Parameters
        ----------
        dynamics : DynamicsFunction
            A callable implementing the ODE for the dynamical system in the form
            f(x, u). Note no explicit time dependency is allowed
        method : Methods, optional
            A enumerator selecting the integration method, by default Methods.RK4
        output : Optional[OutputFunction], optional
            An optional output function applied to each point in the solution of
            the ODE, by default None
        stepsize : ArrayLike, optional
            The integration stepsize(s). If not a single scalar, then must be a
            array of multiple potentially non-uniform stepsizes, and just as
            many control inputs must be passed when the invoking the integrator
        manifold : Manifold, optional
            A product-manifold retraction (see ``example_lib.manifold``). When
            given, the Runge-Kutta stages are formed in tangent coordinates and
            combined through the manifold's ``boxplus`` retraction, so attitude
            quaternions are advanced by the exponential map ``q ⊗ Exp(omega dt)``
            instead of by Euclidean addition. When ``None`` (the default), the
            integrator falls back to flat Euclidean updates, reproducing the
            previous behaviour bit-for-bit.

        Raises
        ------
        NotImplementedError
            `method` is not a member in the Methods enumerator
        ValueError
            `stepsize` is not positive
        """
        self._dynamics = dynamics
        if method not in Methods:
            raise NotImplementedError(f"{method} is not a valid integration method")
        self._method = method
        self._output = output
        self._manifold = manifold
        self._stepsize = jnp.asarray(stepsize)
        if jnp.any(self._stepsize <= 0):
            raise ValueError("Stepsizes must be positive")

    @property
    def stepsize(self):
        """The integration stepsize(s)."""
        return self._stepsize

    @stepsize.setter
    def stepsize(self, value: ArrayLike):
        self._stepsize = jnp.asarray(value)

    def __call__(self, x_op: ArrayLike, u: ArrayLike):
        """Integrates the ODE.

        Parameters
        ----------
        x_op : ArrayLike
            Initial state for the integrator
        u : ArrayLike
            A sequence (array) of control inputs

        Returns
        -------
        Tuple[jax.Array, jax.Array]
            A tuple of the resultant state trajectory and the values of the
            dynamics ODE at each point
        """
        u = jnp.asarray(u)
        if u.ndim == 1:
            stepsize = self._stepsize[0] if self._stepsize.size > 1 else self._stepsize
            _, tup = self._step(x_op, (u, stepsize))
            return tup
        if self._stepsize.size == 1:
            step_fcn = functools.partial(self._step, uniform_dt=self._stepsize)
            _, tup = jax.lax.scan(step_fcn, init=x_op, xs=(u,))
        else:
            _, tup = jax.lax.scan(self._step, init=x_op, xs=(u, self._stepsize))

        return tup

    def _step(self, x_op, tup, uniform_dt=None):
        if uniform_dt is not None:
            stepsize = uniform_dt
            (u,) = tup
        else:
            u, stepsize = tup
        # Work in tangent coordinates: ``rate`` maps a state to its tangent-space
        # velocity and ``boxplus`` retracts a tangent increment onto the manifold.
        # With ``manifold is None`` these collapse to the identity / Euclidean
        # addition, recovering the original flat scheme exactly.
        if self._manifold is None:
            rate = lambda x: self._dynamics(x, u)  # noqa: E731
            boxplus = lambda x, delta: x + delta  # noqa: E731
        else:
            rate = lambda x: self._manifold.to_tangent(x, self._dynamics(x, u))  # noqa: E731
            boxplus = self._manifold.boxplus

        dx = self._dynamics(x_op, u)
        k1 = (
            rate(x_op)
            if self._manifold is None
            else self._manifold.to_tangent(x_op, dx)
        )
        if self._method is Methods.RK4:
            half_stepsize = stepsize / 2
            k2 = rate(boxplus(x_op, half_stepsize * k1))
            k3 = rate(boxplus(x_op, half_stepsize * k2))
            k4 = rate(boxplus(x_op, stepsize * k3))
            increment = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        elif self._method == Methods.EULER:
            increment = k1

        x_op = boxplus(x_op, stepsize * increment)
        y = self._output(x_op) if self._output is not None else x_op
        return x_op, (y, dx)
