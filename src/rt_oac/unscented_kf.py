r"""A manifold Unscented Kalman Filter for the range-only relative-pose system.

Derivative-free alternative to :class:`rt_oac.error_state_ekf.ErrorStateEKF`: instead of
a first-order Jacobian update it propagates 2n+1 sigma points through the *true* nonlinear
dynamics and measurement. This targets the two diagnosed pathologies of the first-order
EKF on this system (carried-estimation findings M0/M1):

* it never linearizes at a (possibly biased) estimate, avoiding the spurious information
  gain in the unobservable subspace a re-linearized EKF accrues during the transient;
* the unscented transform of the range observation samples the curvature of the
  spherical-shell posterior, so it does not manufacture the tangential certainty a single
  range gives -- the source of the EKF position over-confidence (per-block NEES ~118).

It still assumes a Gaussian posterior (a particle filter would represent the shell
exactly), so it mitigates rather than cures, but it is cheap and a near drop-in.

Subclasses ``ErrorStateEKF`` to reuse the manifold step (``_step``), tangent difference
(``_boxminus``), measurement residual (``_residual``), observation (``_observation``) and
the manifold (``_manifold``); the unscented transform on R^3 x SO(3) x R^3 is built from
boxplus / boxminus.
"""

import functools

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla

from rt_oac.error_state_ekf import ErrorStateEKF


class ManifoldUKF(ErrorStateEKF):
    r"""Unscented KF on :math:`\mathbb{R}^3 \times \mathcal{S}^3 \times \mathbb{R}^3`.

    Constructor args are those of :class:`ErrorStateEKF` plus the unscented spread parameters
    ``alpha`` (sigma-point spread), ``beta`` (prior-distribution knowledge; 2 is optimal for
    Gaussians) and ``kappa`` (secondary scaling).
    """

    def __init__(self, *args, alpha=1e-3, beta=2.0, kappa=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        n = self._tangent_dim
        lam = alpha**2 * (n + kappa) - n
        self._gamma = float((n + lam) ** 0.5)
        wm0 = lam / (n + lam)
        wi = 1.0 / (2.0 * (n + lam))
        self._wm = jnp.array([wm0, *([wi] * (2 * n))])
        self._wc = jnp.array([wm0 + (1.0 - alpha**2 + beta), *([wi] * (2 * n))])

    def _sigma_points(self, x, P):
        r"""``2n+1`` ambient sigma points: ``x boxplus {0, +-gamma L[:,i]}``, L = chol(P)."""
        n = self._tangent_dim
        chol = jla.cholesky(P + 1e-12 * jnp.eye(n))  # lower-triangular
        cols = (self._gamma * chol).T  # row i is the i-th column scaled
        deltas = jnp.concatenate([jnp.zeros((1, n)), cols, -cols], axis=0)  # (2n+1, n)
        return jax.vmap(lambda d: self._manifold.boxplus(x, d))(deltas)

    @functools.partial(jax.jit, static_argnames=["self"], donate_argnums=[1, 2])
    def predict(self, x, P, u, dt):
        r"""Unscented predict: propagate the sigma points through one dynamics step.

        The mean is recombined about the propagated central point (one ``boxminus`` iteration);
        the dynamics covariance is the weighted spread, plus the linearized input-noise mapping
        ``G Q_in G^T`` (reused from the EKF) and the optional additive tangent process noise.
        """
        sigmas = self._sigma_points(x, P)
        prop = jax.vmap(lambda s: self._step(s, u, dt))(sigmas)

        ref = prop[0]
        tang = jax.vmap(lambda s: self._boxminus(s, ref))(prop)
        mean = self._manifold.boxplus(ref, self._wm @ tang)
        dx = jax.vmap(lambda s: self._boxminus(s, mean))(prop)
        P_next = jnp.einsum("i,ij,ik->jk", self._wc, dx, dx)

        # Input noise via the EKF's linearized input map (sigma points are over the state only).
        x_next = self._step(x, u, dt)

        def input_map(u_in):
            return self._boxminus(self._step(x, u_in, dt), x_next)

        g_mat = jax.jacobian(input_map)(u)
        P_next += g_mat @ self._in_cov @ g_mat.T
        if self._proc_cov is not None:
            P_next += self._proc_cov
        P_next = 0.5 * (P_next + P_next.T)
        return mean, P_next

    @functools.partial(jax.jit, static_argnames=["self"], donate_argnums=[1, 2])
    def update(self, x, P, y):
        r"""Unscented update with the manifold-aware measurement residual.

        Residuals are taken about the prior predicted measurement ``h(x)`` (so the quaternion
        residual is the manifold log, reusing ``_residual``): ``nu_sig = h(sigma) boxminus h(x)``.
        The innovation covariance ``Pyy``, cross-covariance ``Pxy``, gain ``K = Pxy Pyy^{-1}``, and
        the retracted correction ``x boxplus K (nu_y - nu_mean)`` follow the standard UKF, with
        ``P^+ = P - K Pyy K^T``.
        """
        sigmas = self._sigma_points(x, P)
        # nu_sig[i] = h(sigma_i) boxminus h(x): _residual(x, y') computes y' boxminus h(x).
        nu_sig = jax.vmap(lambda s: self._residual(x, self._observation(s)))(sigmas)
        nu_mean = self._wm @ nu_sig
        dz = nu_sig - nu_mean
        p_yy = jnp.einsum("i,ij,ik->jk", self._wc, dz, dz) + self._obs_cov

        dx = jax.vmap(lambda s: self._boxminus(s, x))(sigmas)
        p_xy = jnp.einsum("i,ij,ik->jk", self._wc, dx, dz)

        gain = jla.solve(p_yy.T, p_xy.T).T  # K = Pxy Pyy^{-1}
        nu_y = self._residual(x, y)  # actual innovation y boxminus h(x)
        x_next = self._manifold.boxplus(x, gain @ (nu_y - nu_mean))
        P_next = P - gain @ p_yy @ gain.T
        P_next = 0.5 * (P_next + P_next.T)
        return x_next, P_next
