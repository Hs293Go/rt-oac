"""Covariance-scheduled trade-off weight (the dual-control coupling resolution).

The observability-vs-standoff trade-off ``weight`` is made a function of the EKF's own
uncertainty: when the estimator is uncertain, down-weight observability so the anchor
dominates (stay conservative, do not out-run the estimator); when it is confident, allow
more observability-seeking. This is the explicit "plan on the belief" answer to the
control/estimation coupling (PROGRESS.md finding #8) -- the controller spends its
observability budget only where the estimator can keep up.

The covariance never enters the JAX graph; only the resulting scalar ``weight`` is
threaded into ``RTController.solve(..., weight=...)``, keeping the schedule plain NumPy.
"""

import numpy as np
from numpy.typing import ArrayLike


def covariance_weight(
    cov: ArrayLike,
    *,
    pos_slice: slice,
    s0: float,
    w_min: float = 0.0,
    w_max: float = 1.0,
    increasing: bool = False,
) -> float:
    """Confidence-scheduled observability weight from ``s = trace(cov[pos_slice])``.

    ``s0`` is the reference (well-conditioned) trace at which the schedule is half-way;
    calibrate it from a stable run. Two conventions for the *same* machinery, chosen by
    whether observability stabilises or destabilises the system:

    * ``increasing=False`` (default, coupling caution): ``w = w_max / (1 + s/s0)`` --
      down-weight observability when uncertain so the anchor dominates and the planner
      does not out-run the estimator (the quadrotor #8 resolution).
    * ``increasing=True`` (active perception): ``w = w_min + (w_max-w_min)*s/(s+s0)`` --
      up-weight observability when uncertain so the agent maneuvers to localise, then
      settles to the anchor once confident (planar: observability is the stabiliser).
    """
    cov = np.asarray(cov)
    s = float(np.trace(cov[pos_slice, pos_slice]))
    w = w_min + (w_max - w_min) * s / (s + s0) if increasing else w_max / (1.0 + s / s0)
    return float(np.clip(w, w_min, w_max))


def covariance_tube_radius(
    cov: ArrayLike,
    *,
    pos_slice: slice,
    s0: float,
    rho_min: float,
    rho_max: float,
) -> float:
    """Tube radius ``rho_min + (rho_max - rho_min) * exp(-s/s0)``.

    The interpretable alternative to :func:`covariance_weight`: tighten the standoff
    tube when the estimator is uncertain (uncertainty -> shrink). Returns a scalar for a
    per-step tube cost/constraint.
    """
    cov = np.asarray(cov)
    s = float(np.trace(cov[pos_slice, pos_slice]))
    return float(rho_min + (rho_max - rho_min) * np.exp(-s / s0))
