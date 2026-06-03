"""Tracking-tube formulations of the tracking/observability trade-off (ported).

Adapted from the tracking worktree's ``balance_constraints``. The smooth squared tube
keeps the path within radius ``rho`` of the reference while the objective maximises
observability; the worktree found this the most robust scheme. Two forms:

* :func:`soft_tube_cost` -- a soft penalty (a ``TrackingCost``) added to the objective.
  This is rt-oac's default tube scheme: it needs no extra constraint slot (the single
  slot is occupied by the inter-vehicle distance constraint) and keeps the objective C1
  for SLSQP.
* :func:`tracking_tube` -- the hard constraint ``||p - p_ref||^2 - rho^2 <= 0`` in
  rt-oac's ``constraint(x0, us, cost)`` convention. Provided for completeness; using it
  requires freeing the single constraint slot (it cannot coexist with the distance
  constraint without multi-constraint support in ``RTController``).

The worktree's ``observability_floor`` (min-eigenvalue >= eps) is deliberately NOT
ported: its Jacobian is non-smooth near degeneracy and the SQP/trust-region solver
converges to dominated solutions (worse than the smooth tube).
"""

import jax.numpy as jnp
from jax.typing import ArrayLike

from . import tracking_cost


def soft_tube_cost(
    rho: float, *, position_indices=(0, 1), w_tube: float = 1.0
) -> tracking_cost.TrackingCost:
    """Soft, smooth tube penalty ``w_tube * sum_t relu(||p_t - p_ref_t||^2 - rho^2)^2``.

    Zero inside the tube, smoothly growing once the path leaves radius ``rho`` -- a
    one-sided, C1 relaxation of the hard tube usable directly as the tracking term in a
    ``WeightedSum`` combiner.
    """
    rho_sq = rho**2
    pos_idx = jnp.asarray(position_indices)

    def cost(xs, _us, p_ref):
        sq = jnp.sum((xs[:, pos_idx] - jnp.asarray(p_ref)) ** 2, axis=1)
        excess = jnp.maximum(sq - rho_sq, 0.0)
        return w_tube * jnp.sum(excess**2)

    return cost


def tracking_tube(rho: float, *, position_indices=(0, 1), p_ref: ArrayLike):
    """Hard tube ``||p(t) - p_ref(t)||^2 - rho^2 <= 0`` (rt-oac ``constraint`` form).

    Returns ``constraint(x0, us, cost) -> per-node vector``; pair with
    ``constraint_bounds=(-inf, 0.0)``. Closes over ``p_ref`` (the rt-oac constraint
    signature carries no reference argument). The squared form is smooth everywhere
    (the bare norm has a ``0/0`` Jacobian when the path sits on the reference).
    """
    rho_sq = rho**2
    pos_idx = jnp.asarray(position_indices)
    p_ref = jnp.asarray(p_ref)

    def constraint(x0, us, cost):
        xs, _ = cost.eval_integrator(x0, us)
        return jnp.sum((xs[:, pos_idx] - p_ref) ** 2, axis=1) - rho_sq

    return constraint
