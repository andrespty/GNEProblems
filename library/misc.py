import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable, Union
import numpy.typing as npt


def flatten_variables(
    vectors: List[npt.NDArray[np.float64]],
    scalars: Union[List[float], npt.NDArray[np.float64]]
) -> npt.NDArray[np.float64]:
    """
    Flatten a list of vectors and optional scalars into a single 1D array.

    This is useful for optimization routines that require all variables
    in a flat array.

    Parameters
    ----------
    vectors : list of numpy.ndarray
        List of 2D arrays (e.g., each shape (n_i, 1)) to flatten.
    scalars : list of float or numpy.ndarray
        Optional list/array of scalar values to append after flattening vectors.

    Returns
    -------
    numpy.ndarray
        1D array containing all flattened vectors followed by the scalars.

    Examples
    --------
    >> vectors = [np.array([[1],[2]]), np.array([[3]])]
    >> scalars = [4, 5]
    >> flatten_variables(vectors, scalars)
    array([1., 2., 3., 4., 5.])
    """
    return np.hstack([v.flatten() for v in vectors] + [scalars])


def deconstruct_vectors(
    vectors: List[npt.NDArray[np.float64]]
) -> npt.NDArray[np.float64]:
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

    Examples
    --------
    >> vectors = [np.array([[1],[2]]), np.array([[3]])]
    >> deconstruct_vectors(vectors)
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






# def check_nash_equillibrium(
#         result: List[np.float64],
#         action_sizes: List[int],
#         player_objective_function: List[int],
#         objective_functions: List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
#         constraints: List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
#         player_constraints: List[List[int]],
#         bounds: List[Tuple[float, float]],
#         paper_res: List[float] = None,
#         epsilon=1e-3
# ):
#     computed_NE = np.array(result).reshape(-1, 1)
#     computed_NE_vectors = construct_vectors(computed_NE, action_sizes)
#     print('Computed Solution: \n', computed_NE_vectors)
#     action_splits = np.cumsum(np.insert(action_sizes, 0, 0))
#
#     # Get the obj function values at the current NE
#     computed_NE_obj_func = objective_check(objective_functions, computed_NE_vectors)
#     print('Computed Objective Function: \n', np.array(computed_NE_obj_func).reshape(-1, 1))
#
#     # Check that constraints are being satisfied
#     if not constraint_check(constraints, computed_NE_vectors):
#         return
#
#     if paper_res is not None:
#         compare_solutions(
#             computed_NE,
#             paper_res,
#             action_sizes,
#             objective_functions,
#         )
#     print('--------------------------------------')
#     # Optimize each player by fixing the opponents
#     for player_idx, p_o_idx in enumerate(player_objective_function):
#         p_var = computed_NE_vectors[player_idx]
#         p_objective = objective_functions[p_o_idx]  # this is a function
#         p_constraints = [constraints[c_idx] for c_idx in player_constraints[p_o_idx] if c_idx]
#
#         wrapped_p_objective = create_wrapped_function(p_objective, computed_NE_vectors, player_idx)
#
#         optimization_constraints = [
#             {'type': 'ineq', 'fun': lambda x: create_wrapped_function(constraint, computed_NE_vectors, player_idx)(x)}
#             for constraint in p_constraints
#         ]
#
#         p_var_0 = np.zeros_like(p_var).flatten()
#         # p_var_0 = p_var.flatten()
#         player_bounds = bounds[action_splits[player_idx]:action_splits[player_idx + 1]]
#         result = minimize(
#             wrapped_p_objective,
#             p_var_0,
#             method='SLSQP',
#             bounds=player_bounds,
#             constraints=optimization_constraints,
#             options={
#                 'disp': False,
#                 'maxiter': 1000,
#                 'ftol': 1e-5
#             }
#         )
#         # Report
#         fixed_vars = computed_NE_vectors[:player_idx] + computed_NE_vectors[player_idx + 1:]
#         opt_actions = np.array(result.x).reshape(-1, 1)
#         new_vars = fixed_vars[:player_idx] + [opt_actions] + fixed_vars[player_idx:]
#         opt_obj_func = objective_check(objective_functions, new_vars)
#         print(f"Player {player_idx + 1}")
#         print(f"Optimized Actions: {result.x}\nOptimized Function value: {result.fun}")
#         print('Optimized Objective Functions: \n', np.array(opt_obj_func).reshape(-1, 1))
#         constraint_check(constraints, new_vars)
#         difference = compare_solutions(
#             computed_NE,
#             deconstruct_vectors(new_vars),
#             action_sizes,
#             objective_functions,
#             solution_name=['Computed', 'Optimized']
#         )
#         print(f"Difference between Objective Functions of Player {player_idx + 1}: {difference[p_o_idx]}")
#         if difference[p_o_idx] > epsilon:
#             print('Not at the NE')
#         else:
#             print(f"Computed Solution is at the NE for Player {player_idx + 1}")
#         print('--------------------------------------')
#     return

