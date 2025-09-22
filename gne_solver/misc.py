from .types import *

def flatten_variables(vectors: VectorList, scalars: List[float]) -> List[float]:
    """
    Flatten a collection of vectors and scalars into a single list.

    Parameters
    ----------
    vectors : VectorList
        List of 2D NumPy arrays with shape (n, 1), representing column
        vectors of floats.
    scalars : List[float]
        A list of scalar values to append after the flattened vectors.

    Returns
    -------
    list of floats
        Single list containing all elements from the input vectors,
        followed by the scalar values.
    Examples
    --------
    >>> import numpy as np
    >>> vectors = [np.array([[1.0], [2.0]]), np.array([[3.0]])]
    >>> scalars = [4.0, 5.0]
    >>> flatten_variables(vectors, scalars)
    >>> [1.0, 2.0, 3.0, 4.0, 5.0]
    """
    return np.hstack([v.flatten() for v in vectors] + [scalars]).tolist()

def deconstruct_vectors(vectors: VectorList) -> Vector:
    return np.concatenate(vectors).reshape(-1, 1)

def print_table(vec1: Vector, vec2: Vector, header1="Vector 1", header2="Vector 2") -> None:
    print(f"{header1:^10} | {header2:^10}")  # Header
    print("-" * 23)
    for v1, v2 in zip(vec1, vec2):
        # Extract scalar value from NumPy arrays
        v1_scalar = v1.item() if isinstance(v1, np.ndarray) else v1
        v2_scalar = v2.item() if isinstance(v2, np.ndarray) else v2
        print(f"{v1_scalar:^10.4f} | {v2_scalar:^10.4f}")  # Align values with 4 decimal places

# Probably delete and no harm is done
def repeat_items(items: List[Union[int, float]], sizes: List[int]) -> np.ndarray:
    return np.array(np.repeat(items, sizes, axis=0).tolist())
