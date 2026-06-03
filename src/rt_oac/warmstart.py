"""Warm-start strategies for the receding-horizon OAC solver.

The companion loop reseeds every solve from the leader reference with the follower
held at hover -- discarding the previous solution. Consecutive receding-horizon
problems are nearly identical, so seeding from the previous solution (shifted one
step) should sharply cut the iteration count, which Phase 0 identified as the
dominant cost (every baseline solve hits the maxiter cap).

Only the follower inputs are decision variables (indices 4..7); the leader columns
are held constant from the reference. So warm-starting replaces only the follower
block of the reference guess with the shifted previous follower solution.
"""

import numpy as np
from numpy.typing import ArrayLike

FOLLOWER_SLICE = slice(4, 8)


def shift_append(seq: ArrayLike) -> np.ndarray:
    """Shift a control sequence one step earlier, repeating the terminal input.

    ``[u0, u1, ..., u_{N-1}] -> [u1, u2, ..., u_{N-1}, u_{N-1}]``. This is the standard
    receding-horizon shift: at the next step the plan's second input becomes the first.
    """
    seq = np.asarray(seq)
    return np.concatenate([seq[1:], seq[-1:]], axis=0)


def warm_guess(
    reference: ArrayLike,
    prev_solution: ArrayLike | None,
    *,
    follower_slice: slice = FOLLOWER_SLICE,
) -> np.ndarray:
    """Build the initial guess for the next solve.

    Parameters
    ----------
    reference
        The reference initial guess ``(window, n_inputs)`` for this step (leader
        inputs from the trajectory, follower at hover); e.g. ``reference_guess(step)``.
    prev_solution
        The previous step's optimized control sequence ``(window, n_inputs)``. When
        ``None`` (first step), the reference is returned unchanged (cold start).
    follower_slice
        Columns that are decision variables and should be warm-started.

    Returns
    -------
    np.ndarray
        ``(window, n_inputs)`` guess: reference leader columns, with the follower
        columns replaced by the shifted previous follower solution.
    """
    reference = np.array(reference, copy=True)
    if prev_solution is None:
        return reference
    prev = np.asarray(prev_solution)
    reference[:, follower_slice] = shift_append(prev[:, follower_slice])
    return reference
