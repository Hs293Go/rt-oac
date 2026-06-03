"""Trajectory/standoff-tracking cost terms (ported from the tracking worktree).

Ported and adapted from ``observability-aware-control-tracking``'s
``observability_aware_control.tracking_cost`` for use in rt-oac's
control/estimation-coupling exercise. A tracking cost is a pure function
``(xs, us, p_ref) -> scalar`` evaluated on the predicted shooting nodes ``xs`` (and
inputs ``us``) produced by the same integrator the observability Gramian uses,
against a per-window position reference ``p_ref``. Threading ``p_ref`` as an explicit
argument (rather than closing over it) lets the optimiser's jitted objective be
compiled once and reused as the reference window slides.
"""

from collections.abc import Callable

import jax.numpy as jnp
from jax.typing import ArrayLike

# (xs, us, p_ref) -> scalar
TrackingCost = Callable[[ArrayLike, ArrayLike, ArrayLike], jnp.ndarray]


def quadratic_tracking_cost(
    *,
    position_indices=(0, 1),
    w_pos: float = 1.0,
    w_ctrl: float = 0.0,
    u_ref: ArrayLike | None = None,
) -> TrackingCost:
    """Builds a quadratic position-tracking (plus optional control-effort) cost.

    Parameters
    ----------
    position_indices : tuple[int, ...]
        State components holding the position tracked against ``p_ref`` (e.g. the
        follower's relative position ``(0, 1, 2)`` for the quadrotor, ``(3, 4)`` for
        the planar leader-follower).
    w_pos : float
        Weight on the summed squared position error.
    w_ctrl : float
        Weight on the summed squared control deviation from ``u_ref`` (or from zero
        when ``u_ref`` is None) -- an optional aggressiveness regulariser.
    u_ref : ArrayLike, optional
        Reference control to penalise deviation from: a single input vector
        (broadcast across the window) or a ``(window, n_inputs)`` array.
    """
    pos_idx = jnp.asarray(position_indices)
    u_ref_arr = None if u_ref is None else jnp.asarray(u_ref)

    def cost(xs, us, p_ref):
        error = xs[:, pos_idx] - jnp.asarray(p_ref)
        value = w_pos * jnp.sum(error**2)
        if w_ctrl:
            du = us if u_ref_arr is None else us - u_ref_arr
            value += w_ctrl * jnp.sum(du**2)
        return value

    return cost
