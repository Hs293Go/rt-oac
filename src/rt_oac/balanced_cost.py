"""Observability cost augmented with a standoff/tracking term (ported + adapted).

Ported from the tracking worktree (``observability-aware-control-tracking``'s
``balanced_cost``) and adapted to rt-oac's controller, for the control/estimation
coupling exercise. ``BalancedCost`` wraps an existing companion
``observability_cost.ObservabilityCost``: it integrates the predicted trajectory
*once* (via the wrapped cost's integrator) and evaluates both the observability
metric and a tracking cost on the same shooting nodes, folding them together with a
swappable :class:`Combiner`. The relative scale between the two disparate objectives
is the scientifically interesting axis (the thesis's open problem), so it is isolated
in the ``Combiner``.

The entry point is ``evaluate(x0, us, dt, p_ref, weight)`` so the controller can thread
``p_ref``/``weight`` as runtime args (compile once, vary the schedule per step). For the
static (Stage A) path, wrap in :class:`AnchoredCost`, which bakes ``p_ref``/``weight``
and exposes the 3-arg ``cost(x0, us, dt)`` call ``RTController`` already uses.
"""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from observability_aware_control import observability_cost

from . import tracking_cost

# (j_track, j_obs, weight) -> scalar; weight threaded at call time.
Combiner = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]


class TrackingCostValue(NamedTuple):
    """Return value of :meth:`BalancedCost.evaluate` (superset of the obs value)."""

    objective: jax.Array
    tracking: jax.Array | None = None
    observability: jax.Array | None = None
    gramians: ArrayLike | None = None
    states: ArrayLike | None = None
    inputs: ArrayLike | None = None


class ObservabilityOnly:
    """``J = j_obs`` (pure observability -- the destabilising baseline)."""

    def __call__(self, j_track, j_obs, weight):
        return j_obs


class TrackingOnly:
    """``J = j_track`` (pure tracking -- the stable anchor)."""

    def __call__(self, j_track, j_obs, weight):
        return j_track


class WeightedSum:
    """``J = j_track + weight * j_obs`` (dimensioned; baseline scalarisation)."""

    def __call__(self, j_track, j_obs, weight):
        return j_track + weight * j_obs


class NormalizedWeightedSum:
    """``J = j_track/s_track + weight * j_obs/s_obs`` (dimensionless ``weight``).

    With ``s_track`` (e.g. ``rho**2 * window``) and ``s_obs`` (observability along the
    reference path), ``weight`` is comparable across experiments -- a direct handle on
    the relative-scale problem.
    """

    def __init__(self, s_track: float, s_obs: float):
        self.s_track = s_track
        self.s_obs = s_obs

    def __call__(self, j_track, j_obs, weight):
        return j_track / self.s_track + weight * j_obs / self.s_obs


class BalancedCost:
    """An observability cost augmented with a trajectory-tracking objective."""

    def __init__(
        self,
        obs_cost: observability_cost.ObservabilityCost,
        tracking: tracking_cost.TrackingCost,
        *,
        combine: Combiner | None = None,
    ):
        self._obs_cost = obs_cost
        self._tracking = tracking
        self._combine = combine if combine is not None else ObservabilityOnly()

    # -- delegate so the controller's constraint helpers reuse the wrapped machinery --
    def eval_integrator(self, x0, us):
        return self._obs_cost.eval_integrator(x0, us)

    def eval_gramian(self, x, u, dt, *args):
        return self._obs_cost.eval_gramian(x, u, dt, *args)

    def evaluate(
        self, x0, us, dt, p_ref, weight: ArrayLike = 1.0, *, return_components=False
    ) -> TrackingCostValue:
        us = jnp.asarray(us)
        xs, _ = self._obs_cost.eval_integrator(x0, us)
        horizon = jnp.broadcast_to(jnp.asarray(dt), us.shape[0])
        gramians = jax.vmap(self._obs_cost.eval_gramian)(xs, us, horizon)
        j_obs = self._obs_cost._gramian_metric(gramians)  # noqa: SLF001
        j_track = self._tracking(xs, us, p_ref)
        objective = self._combine(j_track, j_obs, weight)
        if return_components:
            return TrackingCostValue(objective, j_track, j_obs, gramians, xs, us)
        return TrackingCostValue(objective, j_track, j_obs)

    __call__ = evaluate


class GradNormBalancedCost:
    """Balanced cost that equalises the two objectives' gradient magnitudes.

    The trade-off weight is set to ``weight * ||d j_track/du|| / ||d j_obs/du||``
    (stop-gradient), so observability pulls as hard as tracking regardless of units --
    an automatic answer to the relative-scale problem with no hand-tuned weight.
    """

    def __init__(
        self,
        obs_cost: observability_cost.ObservabilityCost,
        tracking: tracking_cost.TrackingCost,
        *,
        eps: float = 1e-12,
    ):
        self._obs_cost = obs_cost
        self._tracking = tracking
        self._eps = eps

    def eval_integrator(self, x0, us):
        return self._obs_cost.eval_integrator(x0, us)

    def eval_gramian(self, x, u, dt, *args):
        return self._obs_cost.eval_gramian(x, u, dt, *args)

    def _terms(self, x0, us, dt, p_ref):
        xs, _ = self._obs_cost.eval_integrator(x0, us)
        horizon = jnp.broadcast_to(jnp.asarray(dt), us.shape[0])
        gramians = jax.vmap(self._obs_cost.eval_gramian)(xs, us, horizon)
        j_obs = self._obs_cost._gramian_metric(gramians)  # noqa: SLF001
        j_track = self._tracking(xs, us, p_ref)
        return j_track, j_obs

    def evaluate(
        self, x0, us, dt, p_ref, weight: ArrayLike = 1.0, *, return_components=False
    ) -> TrackingCostValue:
        us = jnp.asarray(us)
        grad_track = jax.grad(lambda u: self._terms(x0, u, dt, p_ref)[0])(us)
        grad_obs = jax.grad(lambda u: self._terms(x0, u, dt, p_ref)[1])(us)
        scale = jnp.linalg.norm(grad_track) / (jnp.linalg.norm(grad_obs) + self._eps)
        scale = jax.lax.stop_gradient(weight * scale)
        j_track, j_obs = self._terms(x0, us, dt, p_ref)
        objective = j_track + scale * j_obs
        if return_components:
            xs, _ = self._obs_cost.eval_integrator(x0, jnp.asarray(us))
            return TrackingCostValue(objective, j_track, j_obs, None, xs, us)
        return TrackingCostValue(objective, j_track, j_obs)

    __call__ = evaluate


class AnchoredCost:
    """3-arg adapter: bakes ``p_ref``/``weight`` so ``cost(x0, us, dt)`` works.

    This is the Stage-A (static) path -- ``RTController`` calls the wrapped cost with
    three args, exactly as it calls a plain ``ObservabilityCost``.
    """

    def __init__(self, balanced, *, p_ref: ArrayLike, weight: ArrayLike = 1.0):
        self._balanced = balanced
        self._p_ref = jnp.asarray(p_ref)
        self._weight = weight

    def eval_integrator(self, x0, us):
        return self._balanced.eval_integrator(x0, us)

    def eval_gramian(self, x, u, dt, *args):
        return self._balanced.eval_gramian(x, u, dt, *args)

    def __call__(self, x0, us, dt) -> TrackingCostValue:
        return self._balanced.evaluate(x0, us, dt, self._p_ref, self._weight)


def make_balanced(
    obs_cost: observability_cost.ObservabilityCost,
    tracking: tracking_cost.TrackingCost,
    *,
    scheme: str,
    s_track: float = 1.0,
    s_obs: float = 1.0,
) -> BalancedCost | GradNormBalancedCost:
    """Builds the (un-anchored) balanced cost for ``scheme``.

    ``scheme`` in {observability_only, tracking_only, weighted, tube, normalized,
    gradnorm}. ``tube`` reuses the ``weighted`` combiner with a soft-tube tracking term
    (see :func:`rt_oac.balance_constraints.soft_tube_cost`). For Stage A, wrap it
    in :class:`AnchoredCost`; for Stage B, pass it to
    ``RTController.solve(p_ref=..., weight=...)``.
    """
    if scheme == "gradnorm":
        return GradNormBalancedCost(obs_cost, tracking)
    if scheme == "observability_only":
        combine: Combiner = ObservabilityOnly()
    elif scheme == "tracking_only":
        combine = TrackingOnly()
    elif scheme in {"weighted", "tube"}:
        combine = WeightedSum()
    elif scheme == "normalized":
        combine = NormalizedWeightedSum(s_track, s_obs)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    return BalancedCost(obs_cost, tracking, combine=combine)
