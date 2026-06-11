from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def complementary_indices(size: int, ids: ArrayLike) -> jax.Array:
    """Finds the complement of given indices in a index sequence.

    Parameters
    ----------
    size : int
        Size of the index sequence
    ids : ArrayLike
        Index whose complement is to be taken

    Returns
    -------
    jax.Array
        Array containing the complement of ids
    """
    id_complement = jnp.setdiff1d(jnp.arange(0, size), ids)
    return id_complement


@jax.jit
def separate_array(
    mat: jax.Array, idx: ArrayLike, idy: ArrayLike | None = None
) -> tuple[ArrayLike, ArrayLike]:
    """Extracts the idx and idy-th components (along the last axis) of the input matrix.

    Parameters
    ----------
    mat : jax.Array
        Input array to be separated
    idx : ArrayLike
        First components of the array to be separated out
    idy : Optional[ArrayLike], optional
        Second components of the array to be separated out, by default None, in which
        case said components will be the complement of idx

    Returns
    -------
    Tuple[ArrayLike, ArrayLike]
        The idx and idy-th components of the input matrix respectively
    """
    if idy is None:
        idy = complementary_indices(mat.shape[-1], idx)
    return mat[..., idx], mat[..., idy]


@jax.jit
def combine_array(
    m_1: ArrayLike,
    m_2: ArrayLike,
    idx: ArrayLike,
    idy: ArrayLike,
) -> jax.Array:
    """Builds an array by combining idx and idy-th components (along the last axis) with
    values m_1 and m_2. Inverse of separate_array.

    Parameters
    ----------
    shape : Union[int, Sequence[int]]
        The shape of the array to be built
    m_1 : ArrayLike
        Values of the idx-th components
    m_2 : ArrayLike
        Values of the idy-th components
    idx : ArrayLike
        First components
    idy : ArrayLike
        Second components

    Returns
    -------
    jax.Array
        The built array
    """
    m = jnp.empty((len(m_1), len(idx) + len(idy)))

    m = m.at[..., idx].set(m_1)
    m = m.at[..., idy].set(m_2)
    return m


def separate_array_argument(
    fun: Callable,
    shape: int | Sequence[int],
    idx: ArrayLike,
    idy: ArrayLike | None = None,
) -> Callable:
    """Wraps a function taking an array as the first argument to give a function taking
    the 'idx' and 'idy' components of said array along the last axis as the first and
    second arguments respectively.

    Parameters
    ----------
    fun : Callable
        A function taking an array as its first argument to be wrapped
    shape : Union[int, Sequence[int]]
        Size of the array to be used as the first argument
    idx : ArrayLike
        Components of the array to be passed in the first argument of the resulting
        function
    idy : Optional[ArrayLike], optional
        Components of the array to be passed in the second argument of the resulting
        function , by default None, in which case said components will be the
        complement of idx

    Returns
    -------
    _type_
        _description_
    """
    if idy is None:
        idy = complementary_indices(shape if isinstance(shape, int) else shape[-1], idx)

    def wrapped(m_1, m_2, *args, **kwargs):
        m = combine_array(m_1, m_2, idx, idy)
        return fun(m, *args, **kwargs)

    return wrapped
