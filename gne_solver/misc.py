import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable, Union
import numpy.typing as npt


def flatten_variables(vectors,scalars):
    """
    Flatten a collection of vectors and scalars into a single list.

    This utility is typically used in optimization routines that require
    all decision variables to be represented as a flat one-dimensional array.

    Parameters
    ----------
    vectors : list of numpy.ndarray
        List of one- or two-dimensional arrays (commonly column vectors
        with shape ``(n_i, 1)``) to be flattened.
    scalars : list of float or numpy.ndarray
        Scalars to append after the flattened vectors. Can be provided
        either as a Python list of floats or as a NumPy array of shape ``(m,)``.

    Returns
    -------
    list of float
        A flat list containing all vector entries followed by the scalars.

    Examples
    --------
    >>> import numpy as np
    >>> vectors = [np.array([[1.0], [2.0]]), np.array([[3.0]])]
    >>> scalars = [4.0, 5.0]
    >>> flatten_variables(vectors, scalars)
    [1.0, 2.0, 3.0, 4.0, 5.0]
    """
    return np.hstack([v.flatten() for v in vectors] + [scalars]).tolist()


def deconstruct_vectors(vectors):
    """
    Concatenate a list of 2D arrays into a single column vector.

    Useful for converting per-player action arrays into a single optimization
    vector.

    Parameters
    ----------
    vectors : list of numpy.ndarray
        List of 2D arrays (shape (n_i, 1) each).

    Returns
    -------
    numpy.ndarray
        2D column vector of shape (sum(n_i), 1) containing all elements
        concatenated vertically.

    See Also: [`gne_solver.misc.flatten_variables`](gne_solver.misc.flatten_variables) LINK

    Examples
    --------
    >>> vectors = [np.array([[1],[2]]), np.array([[3]])]
    >>> deconstruct_vectors(vectors)
    array([[1],
           [2],
           [3]])
    """
    return np.concatenate(vectors).reshape(-1, 1)


def print_table(vec1, vec2, header1="Vector 1", header2="Vector 2"):
    """
    Print two vectors side by side in a formatted table.

    This function aligns values in columns and prints them with 4 decimal places.
    It works with lists, NumPy arrays, or a mix of both.

    Parameters
    ----------
    vec1 : list or numpy.ndarray
        First vector to print. Can contain scalars or 1-element arrays.
    vec2 : list or numpy.ndarray
        Second vector to print. Must have the same length as `vec1`.
    header1 : str, optional
        Column header for the first vector (default is "Vector 1").
    header2 : str, optional
        Column header for the second vector (default is "Vector 2").

    Returns
    -------
    None
        Prints the table to standard output.

    Examples
    --------
    >> vec1 = [1, 2, 3]
    >> vec2 = np.array([[1.1], [2.2], [3.3]])
    >> print_table(vec1, vec2, header1="A", header2="B")
         A      |     B
    -----------------------
       1.0000 |   1.1000
       2.0000 |   2.2000
       3.0000 |   3.3000
    """
    print(f"{header1:^10} | {header2:^10}")  # Header
    print("-" * 23)
    for v1, v2 in zip(vec1, vec2):
        # Extract scalar value from NumPy arrays
        v1_scalar = v1.item() if isinstance(v1, np.ndarray) else v1
        v2_scalar = v2.item() if isinstance(v2, np.ndarray) else v2
        print(f"{v1_scalar:^10.4f} | {v2_scalar:^10.4f}")  # Align values with 4 decimal places

def repeat_items(items: List[Union[int, float]], sizes: List[int]) -> np.ndarray:
    """
    Repeat each item in a list according to corresponding sizes.

    Parameters
    ----------
    items : list of int or float
        The list of items to be repeated, e.g., [1, 2, 3].
    sizes : list of int
        The number of times to repeat each corresponding item in `items`.
        Must have the same length as `items`.

    Returns
    -------
    numpy.ndarray
        1D array where each item is repeated according to `sizes`.

    Examples
    --------
    >> items = [1, 2, 3, 4, 5]
    >> sizes = [2, 3, 1, 4, 1]
    >> repeat_items(items, sizes)
    array([1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 5])
    """
    return np.array(np.repeat(items, sizes, axis=0).tolist())
