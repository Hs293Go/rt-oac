"""Observability objective metrics (and smooth surrogates for the min-eigenvalue).

A metric maps a stack of per-node observability Gramians ``(N, m, m)`` (symmetric
PSD) to a scalar the controller *minimizes*. The companion baseline maximizes the
worst-direction precision via the minimum singular value -- but that objective is
non-smooth and, at the operating points seen here, *singular* (min-eigenvalue ~ 0),
so its gradient is ~0 and the optimizer cannot make progress (Phase 0 finding). The
surrogates below are smooth and have well-defined nonzero gradients even when the
Gramian is near-singular, so a lean solver can actually climb observability.

All metrics are written so that **smaller is better** (maximizing the underlying
observability quantity), matching ``scipy.optimize.minimize``. Use symmetric
``eigvalsh`` (the Gramian is symmetric PSD) rather than SVD -- cheap, differentiable.
"""

import jax.numpy as jnp
import jax.scipy.special as jsp
from jax.typing import ArrayLike

_EPS = 1e-3


def _eigs(gramians: ArrayLike) -> jnp.ndarray:
    """Eigenvalues of each symmetric Gramian, shape ``(N, m)``, clipped at 0."""
    g = jnp.asarray(gramians)
    g = 0.5 * (g + jnp.swapaxes(g, -1, -2))
    return jnp.maximum(jnp.linalg.eigvalsh(g), 0.0)


def reciprocal_min_eig(gramians: ArrayLike, eps: float = _EPS) -> jnp.ndarray:
    """Baseline: ``eps / (sum_k lambda_min(W_k) + eps)`` (non-smooth, can be singular).

    Equivalent to the companion ``_default_gramian_metric`` but via ``eigvalsh``.
    """
    smallest = _eigs(gramians)[:, 0]
    return eps / (smallest.sum() + eps)


def neg_logdet(gramians: ArrayLike, eps: float = _EPS) -> jnp.ndarray:
    """D-optimality surrogate: ``-sum_k sum_i log(lambda_i(W_k) + eps)``.

    Smooth everywhere; maximizes the log-volume of the information ellipsoid.
    Caveat: can inflate already-observable directions while leaving the worst one
    weak -- guarded by the true-objective / EKF acceptance tests.
    """
    return -jnp.log(_eigs(gramians) + eps).sum()


def neg_softmin_eig(gramians: ArrayLike, tau: float = 1.0) -> jnp.ndarray:
    """Soft-min surrogate: ``-sum_k softmin_tau(eig(W_k))``.

    ``softmin_tau(l) = -tau * logsumexp(-l / tau)`` is a smooth lower bound on
    ``min_i l_i`` (recovered as ``tau -> 0``). Maximizing it directly targets the worst
    observable direction like the baseline, but with a usable gradient at degeneracy.
    """
    eigs = _eigs(gramians)
    softmin = -tau * jsp.logsumexp(-eigs / tau, axis=1)
    return -softmin.sum()


def neg_harmonic_min_eig(gramians: ArrayLike, eps: float = _EPS) -> jnp.ndarray:
    """Smooth worst-direction surrogate: ``-sum_k 1 / sum_i 1/(lambda_i + eps)``.

    The harmonic mean of eigenvalues is dominated by (and a smooth proxy for) the
    smallest eigenvalue, while being differentiable. ``1/trace(W^{-1})``-style,
    A-optimality-flavored.
    """
    eigs = _eigs(gramians)
    harmonic = 1.0 / (1.0 / (eigs + eps)).sum(axis=1)
    return -harmonic.sum()


def neg_sum_k_smallest(gramians: ArrayLike, k: int = 2) -> jnp.ndarray:
    """Maximize the sum of the ``k`` smallest eigenvalues per node (crossing-robust)."""
    return -_eigs(gramians)[:, :k].sum()


#: Registry for config-driven selection.
METRICS = {
    "reciprocal_min_eig": reciprocal_min_eig,
    "neg_logdet": neg_logdet,
    "neg_softmin_eig": neg_softmin_eig,
    "neg_harmonic_min_eig": neg_harmonic_min_eig,
    "neg_sum_k_smallest": neg_sum_k_smallest,
}
