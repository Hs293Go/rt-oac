r"""A quaternion / manifold-aware error-state EKF for the relative-pose drone system.

The placeholder :class:`example_lib.misc.simple_ekf.SimpleEKF` treats the state as a
flat :math:`\\mathbb{R}^{10}` vector: it forms the innovation by Euclidean subtraction
and corrects by Euclidean addition. On the unit-quaternion attitude factor that is only
first-order correct -- the corrected quaternion leaves :math:`\\mathcal{S}^3`, the
covariance carries a spurious radial gauge, and the filter grows overconfident as the
attitude error accumulates (we observed a ~7 m terminal position error against a 0.6 m
3-sigma envelope, with the follower flying out to 16 m).

This module implements the standard *error-state* (a.k.a. *multiplicative*) EKF instead.
The ambient state ``x`` lives on the product manifold

.. math::

    \\mathcal{M} = \\mathbb{R}^3 \\times \\mathcal{S}^3 \\times \\mathbb{R}^3,

with a 9-dimensional tangent space. The mean is carried as the ambient point ``x`` and
the covariance ``P`` as a 9x9 Gramian in the *tangent* coordinates of
``example_lib.models.inter_quadrotor_pose.MANIFOLD``. All filter algebra -- transition
Jacobian, measurement Jacobian, gain, covariance update -- is done in those 9-d tangent
coordinates, and every correction is retracted onto the manifold through ``boxplus``
(:math:`q \\leftarrow q \\otimes \\mathrm{Exp}(\\delta\\theta)`), so the quaternion
stays unit-norm by construction.

Conventions (validated against the model's own ``observation``/``dynamics`` and the
``example_lib.math`` quaternion helpers):

* quaternions are scalar-last (``xyzw``);
* the retraction is the *right* perturbation ``q ⊗ Exp(xi)`` with ``xi`` a body-frame
  rotation vector -- exactly :func:`example_lib.math.quaternion_exp`;
* hence the tangent difference (``boxminus``) of two attitudes is
  ``Log(q_a^{-1} ⊗ q_b)`` (:func:`example_lib.math.quaternion_log`), which round-trips
  ``boxplus`` to float64 precision (unit-tested);
* the measurement residual is therefore the manifold log
  ``Log(q_hat^{-1} ⊗ q_meas)`` for attitude and ``y_range - h_range(x_hat)`` for the
  squared range, **never** a raw quaternion subtraction.

The measurement model is taken verbatim from the model
(:func:`example_lib.models.inter_quadrotor_pose.observation`,
``h(x) = [||r||^2, q]``), so the range convention (full squared norm, not one-half) and
the quaternion ordering match the simulated measurements automatically.
"""

import functools

from example_lib import math
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
from observability_aware_control import integrator

__all__ = ["ErrorStateEKF"]


class ErrorStateEKF:
    r"""Error-state (multiplicative) EKF on a Euclidean x SO(3) x Euclidean manifold.

    The state manifold is :math:`\mathbb{R}^3 \times \mathcal{S}^3 \times \mathbb{R}^3`.
    The mean is the ambient state ``x`` (size ``manifold.ambient_dim``, 10 here); the
    covariance ``P`` is a ``tangent_dim x tangent_dim`` (9x9) Gramian in the manifold's
    tangent coordinates. ``predict`` propagates the mean by one manifold integrator step
    and the covariance through the linearization of that step in tangent coordinates;
    ``update`` forms the innovation and Jacobian with manifold-aware residuals and
    retracts the correction through ``boxplus``.

    Parameters
    ----------
    dynamics : Callable[[Array, Array], Array]
        The continuous-time ambient dynamics ``f(x, u)`` (ambient-rate valued).
    observation : Callable[[Array], Array]
        The ambient measurement map ``h(x) -> [range, q(4)]`` (size 5).
    manifold : example_lib.manifold.Manifold
        The product manifold; supplies ``boxplus``, ``to_tangent`` and the tangent /
        ambient dimensions. The attitude factor is assumed to be the single
        ``SO3Block`` occupying ambient indices ``[3:7]`` and tangent indices ``[3:6]``.
    in_cov : Array
        Input/process noise covariance ``Q`` (``n_inputs x n_inputs``); mapped into the
        tangent space through ``G = d(boxplus step)/du``.
    obs_cov : Array
        Measurement noise covariance for the *residual* space ``R`` (4x4):
        ``diag(range_var, att_var, att_var, att_var)``.
    proc_cov : Array, optional
        An additive *tangent* process-noise covariance ``Q_t`` (``tangent_dim x
        tangent_dim``, 9x9) added to ``P`` each predict, on top of the input-mapped
        ``G Q G^T``. The input mapping alone only injects uncertainty through the
        commanded-input channels; in the closed loop the controller drives the drones to
        large baselines where the squared-range measurement weakly observes the
        tangential position, so without a small process-noise floor the covariance
        collapses and the filter turns overconfident (the error grows outside a shrunk
        3-sigma envelope). A modest ``Q_t`` keeps the filter honest -- the standard
        remedy for un-modeled dynamics. When ``None`` (default), only ``G Q G^T`` is
        used.
    method : observability_aware_control.integrator.Methods, optional
        Integration method for the mean propagation (default EULER, matching the
        closed-loop simulator).
    """

    def __init__(
        self,
        dynamics,
        observation,
        manifold,
        in_cov,
        obs_cov,
        proc_cov=None,
        method=None,
    ):
        self._dynamics = dynamics
        self._observation = observation
        self._manifold = manifold
        self._in_cov = jnp.asarray(in_cov)
        self._obs_cov = jnp.asarray(obs_cov)
        self._proc_cov = None if proc_cov is None else jnp.asarray(proc_cov)
        self._method = method if method is not None else integrator.Methods.EULER

        self._tangent_dim = manifold.tangent_dim

        # Ambient slice of the single SO(3) attitude factor (used to splice the
        # quaternion out of the ambient state for the manifold log). Resolved from the
        # public block layout -- accumulate each block's ``ambient_size`` until the
        # first block whose tangent DOF is smaller than its ambient size, i.e. the
        # quaternion factor (3 tangent DOF vs 4 ambient components). For this model the
        # manifold is [Euclidean(3), SO3, Euclidean(3)] so this is ambient [3:7].
        ambient = 0
        self._q_ambient = None
        for block in manifold.blocks:
            if block.tangent_size < block.ambient_size:
                self._q_ambient = slice(ambient, ambient + block.ambient_size)
                break
            ambient += block.ambient_size
        if self._q_ambient is None:
            raise ValueError("manifold has no SO(3)/quaternion attitude factor")

    @property
    def in_cov(self):
        """Input/process noise covariance ``Q``."""
        return self._in_cov

    @property
    def obs_cov(self):
        """Measurement (residual-space) noise covariance ``R``."""
        return self._obs_cov

    # ---- mean propagation ---------------------------------------------------------
    def _step(self, x, u, dt):
        """One manifold integrator step ``x -> x^+`` (attitude via the exp map).

        Implements ``boxplus(x, dt * to_tangent(x, f(x, u)))`` for EULER and a
        tangent-space RK4 otherwise -- the same scheme as
        :class:`observability_aware_control.integrator.Integrator` with a manifold,
        but written here so it is differentiable in ``(x, u)`` without re-creating an
        ``Integrator`` inside the jitted closure.
        """
        boxplus = self._manifold.boxplus

        def rate(state):
            return self._manifold.to_tangent(state, self._dynamics(state, u))

        k1 = rate(x)
        if self._method is integrator.Methods.RK4:
            half = dt / 2.0
            k2 = rate(boxplus(x, half * k1))
            k3 = rate(boxplus(x, half * k2))
            k4 = rate(boxplus(x, dt * k3))
            increment = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        else:
            increment = k1
        return boxplus(x, dt * increment)

    # ---- manifold residuals -------------------------------------------------------
    def _boxminus(self, x_a, x_b):
        r"""Tangent difference ``x_a \\boxminus x_b`` (inverse of ``boxplus``).

        Euclidean blocks subtract; the attitude block returns
        ``Log(q_b^{-1} ⊗ q_a)`` so that ``boxplus(x_b, boxminus(x_a, x_b)) == x_a``.
        """
        diff = x_a - x_b  # correct for every Euclidean component
        qa = x_a[self._q_ambient]
        qb = x_b[self._q_ambient]
        dtheta = math.quaternion_log(
            math.quaternion_product(math.quaternion_inverse(qb), qa)
        )
        # Assemble the 9-vector tangent difference: Euclidean parts straight through,
        # attitude part replaced by the 3-vector rotation vector.
        tangent = jnp.concatenate([
            diff[0:3],
            dtheta,
            diff[7:10],
        ])
        return tangent

    def _residual(self, x, y):
        r"""Measurement residual ``nu = y \\boxminus h(x)`` in the 4-d residual space.

        ``range`` residual ``y_range - h_range(x)``; attitude residual
        ``Log(q_hat^{-1} ⊗ q_meas)`` -- the manifold log, not a quaternion subtraction.
        """
        hx = self._observation(x)
        range_resid = y[0:1] - hx[0:1]
        q_hat = hx[1:5]
        q_meas = y[1:5]
        att_resid = math.quaternion_log(
            math.quaternion_product(math.quaternion_inverse(q_hat), q_meas)
        )
        return jnp.concatenate([range_resid, att_resid])

    # ---- predict ------------------------------------------------------------------
    @functools.partial(jax.jit, static_argnames=["self"], donate_argnums=[1, 2])
    def predict(self, x, P, u, dt):
        r"""Propagate ``(x, P)`` one step under input ``u`` over ``dt``.

        The mean advances by one manifold integrator step. The error-state transition
        ``F`` (9x9) is the Jacobian w.r.t. a tangent perturbation ``delta`` at
        ``delta = 0`` of

        .. math::

            \\delta \\mapsto \\mathrm{step}(x \\boxplus \\delta, u, dt)
                              \\boxminus \\mathrm{step}(x, u, dt),

        and the input/noise mapping ``G = d(step)/du \\boxminus`` is the Jacobian of the
        same propagated state w.r.t. ``u``. The covariance update is
        ``P^+ = F P F^T + G Q G^T (+ Q_t)``, with the optional additive tangent
        process-noise floor ``Q_t`` (see ``proc_cov``).

        Parameters
        ----------
        x : Array
            Ambient state (size 10).
        P : Array
            Tangent covariance (9x9).
        u : Array
            Input (size 8).
        dt : float
            Step size.

        Returns
        -------
        tuple[Array, Array]
            The propagated ``(x^+, P^+)``.
        """
        x_next = self._step(x, u, dt)

        def err_transition(delta):
            x_pert = self._manifold.boxplus(x, delta)
            return self._boxminus(self._step(x_pert, u, dt), x_next)

        def input_map(u_in):
            return self._boxminus(self._step(x, u_in, dt), x_next)

        zero = jnp.zeros(self._tangent_dim)
        F = jax.jacobian(err_transition)(zero)
        G = jax.jacobian(input_map)(u)

        P_next = F @ P @ F.T + G @ self._in_cov @ G.T
        if self._proc_cov is not None:
            P_next += self._proc_cov
        # Symmetrize to suppress round-off asymmetry that can grow over a long run.
        P_next = 0.5 * (P_next + P_next.T)
        return x_next, P_next

    # ---- update -------------------------------------------------------------------
    @functools.partial(jax.jit, static_argnames=["self"], donate_argnums=[1, 2])
    def update(self, x, P, y):
        r"""Correct ``(x, P)`` with measurement ``y = [range, q(4)]`` (size 5).

        The innovation ``nu`` (size 4) and Jacobian ``H`` (4x9) are formed with
        manifold-aware residuals; the gain is ``K = P H^T (H P H^T + R)^{-1}``; the mean
        correction is retracted as ``x^+ = boxplus(x, K nu)`` (keeping ``q`` unit-norm);
        the covariance uses the Joseph form
        ``P^+ = (I - K H) P (I - K H)^T + K R K^T``.

        Parameters
        ----------
        x : Array
            Ambient state (size 10).
        P : Array
            Tangent covariance (9x9).
        y : Array
            Measurement ``[range, q(4)]`` (size 5).

        Returns
        -------
        tuple[Array, Array]
            The corrected ``(x^+, P^+)``.
        """
        nu = self._residual(x, y)

        # H = d(residual)/d(delta) at delta = 0. The residual depends on the estimate
        # only through h(x boxplus delta); differentiating it (rather than h alone)
        # bakes the manifold log's tangent map into H so H and nu live in the same
        # 4-d residual space.
        def residual_of_delta(delta):
            return self._residual(self._manifold.boxplus(x, delta), y)

        zero = jnp.zeros(self._tangent_dim)
        # The residual is y boxminus h(x boxplus delta); its derivative w.r.t delta is
        # the negative of the measurement Jacobian, which is what the gain expects.
        H = jax.jacobian(residual_of_delta)(zero)
        H = -H  # H := d h_tangent / d delta (residual = nu0 - H delta + ...)

        S = H @ P @ H.T + self._obs_cov
        K = jla.solve(S.T, H @ P).T  # K = P H^T S^{-1}

        x_next = self._manifold.boxplus(x, K @ nu)

        identity = jnp.eye(self._tangent_dim)
        ikh = identity - K @ H
        P_next = ikh @ P @ ikh.T + K @ self._obs_cov @ K.T
        P_next = 0.5 * (P_next + P_next.T)
        return x_next, P_next
