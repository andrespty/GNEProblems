import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable, Union
import numpy.typing as npt
from .misc import *

def construct_vectors(
    actions: npt.NDArray[np.float64],
    action_sizes: List[int]
) -> List[npt.NDArray[np.float64]]:
    """
    Split a concatenated action array into separate action vectors for each player.

    This function takes a single concatenated array of all players' actions and
    splits it into a list of arrays based on the provided sizes for each player's
    action vector.

    Parameters
    ----------
    actions : numpy.ndarray of shape (sum(action_sizes), 1)
        A 2D NumPy array containing all players' actions stacked vertically.
        The number of rows must equal the sum of all entries in ``action_sizes``.
    action_sizes : list of int
        A list specifying the length of each player's action vector.
        The sum of these sizes must match the number of rows in ``actions``.

    Returns
    -------
    list of numpy.ndarray
        A list of 2D NumPy arrays, each corresponding to one player's action vector.
        The arrays are in the same order as the players in ``action_sizes``.

    Examples
    --------
    >> actions = np.array([[1.0], [2.0], [3.0], [4.0]])
    >> action_sizes = [2, 2]
    >> construct_vectors(actions, action_sizes)
    [array([[1.],
            [2.]]),
     array([[3.],
            [4.]])]
    """
    value_array = np.array(actions)
    indices = np.cumsum(action_sizes)
    return np.split(value_array, indices[:-1])

def create_wrapped_function(
    original_func: Callable[[List[npt.NDArray[np.float64]]], npt.NDArray[np.float64]],
    vars: List[npt.NDArray[np.float64]],
    player_idx: int
) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """
    Create a function wrapper with all but one player's variables fixed.

    This function returns a new function where all players' variables are fixed
    except for the player at the given index. The returned function takes only
    that player's variable vector as input, reconstructs the full variable list
    (including the fixed ones), and calls the original function.

    Parameters
    ----------
    original_func : Callable[[list of numpy.ndarray], numpy.ndarray]
        The original function that operates on a list of player variables.
        Each element of the list is a 2D NumPy array representing a player's
        action or decision variables.
    vars : list of numpy.ndarray
        The full list of player variables to be passed to ``original_func``.
        Each array should be shaped as (n, 1).
    player_idx : int
        The index of the player whose variable will remain free (not fixed).
        Must be between 0 and ``len(vars) - 1``.

    Returns
    -------
    Callable[[numpy.ndarray], numpy.ndarray]
        A wrapper function that accepts a single NumPy array for the chosen player's
        variables (shape (n, 1) or flattenable to that) and returns the output of
        ``original_func`` with all variables assembled.

    Examples
    --------
    >> def sum_all(players):
    ...     return sum(p.sum() for p in players)
    >> vars = [np.array([[1.0]]), np.array([[2.0]])]
    >> wrapped = create_wrapped_function(sum_all, vars, player_idx=0)
    >> wrapped([[3.0]])
    5.0
    """
    fixed_vars = vars[:player_idx] + vars[player_idx + 1:]  # list of np vectors

    def wrap_func(player_var_opt):
        player_var_opt = np.array(player_var_opt).reshape(-1, 1)
        new_vars = fixed_vars[:player_idx] + [player_var_opt] + fixed_vars[player_idx:]
        return original_func(new_vars)

    return wrap_func

def create_wrapped_function_single(
    original_func: Callable[[List[npt.NDArray[np.float64]]], npt.NDArray[np.float64]],
    vars: List[npt.NDArray[np.float64]],
    player_idx: int
) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """
    Create a function wrapper with all but one player's variables fixed.

    This function returns a new function where all players' variables are fixed
    except for the player at the given index. The returned function takes only
    that player's variable vector as input, reconstructs the full variable list
    (including the fixed ones), and calls the original function.

    Parameters
    ----------
    original_func : Callable[[list of numpy.ndarray], numpy.ndarray]
        The original function that operates on a list of player variables.
        Each element of the list is a 2D NumPy array representing a player's
        action or decision variables.
    vars : list of numpy.ndarray
        The full list of player variables to be passed to ``original_func``.
        Each array should be shaped as (n, 1).
    player_idx : int
        The index of the player whose variable will remain free (not fixed).
        Must be between 0 and ``len(vars) - 1``.

    Returns
    -------
    Callable[[numpy.ndarray], numpy.ndarray]
        A wrapper function that accepts a single NumPy array for the chosen player's
        variables (shape (n, 1) or flattenable to that) and returns the output of
        ``original_func`` with all variables assembled.

    Examples
    --------
    >> def sum_all(players):
    ...     return sum(p.sum() for p in players)
    >> vars = [np.array([[1.0]]), np.array([[2.0]])]
    >> wrapped = create_wrapped_function(sum_all, vars, player_idx=0)
    >> wrapped([[3.0]])
    5.0
    """
    fixed_vars = vars[:player_idx] + vars[player_idx + 1:]  # list of np vectors

    def wrap_func(player_var_opt):
        player_var_opt = np.array(player_var_opt).reshape(-1, 1)
        new_vars = fixed_vars[:player_idx] + [player_var_opt] + fixed_vars[player_idx:]
        return original_func(new_vars)[player_idx]

    return wrap_func

def objective_check(
    objective_functions: List[Callable[[List[npt.NDArray[np.float64]]], npt.NDArray[np.float64]]],
    actions: List[npt.NDArray[np.float64]]
) -> npt.NDArray[np.float64]:
    """
    Evaluate multiple objective functions on a set of player actions.

    Parameters
    ----------
    objective_functions : list of callables
        Each callable should take a list of NumPy arrays (one per player)
        and return a NumPy array of objective values (shape (m, 1) or flattenable to that).
    actions : list of numpy.ndarray
        List of 2D NumPy arrays, each representing a player's action vector.
        All arrays should be shaped as (n, 1).

    Returns
    -------
    numpy.ndarray
        A single 2D array of shape (total_objectives, 1), formed by concatenating
        the outputs of each objective function vertically.

    Examples
    --------
    >> def obj1(actions): return np.array([[actions[0].sum()]])
    >> def obj2(actions): return np.array([[actions[1].sum()]])
    >> actions = [np.array([[1.0]]), np.array([[2.0]])]
    >> objective_check([obj1, obj2], actions)
    array([[1.],
           [2.]])
    """
    objective_values = []
    for objective in objective_functions:
        o = objective(actions)
        objective_values.append(o)
    return np.concatenate(objective_values).reshape(-1, 1)

def constraint_check(
    constraints: List[Callable[[List[npt.NDArray[np.float64]]], npt.NDArray[np.float64]]],
    actions: List[npt.NDArray[np.float64]],
    epsilon: float = 1e-3
) -> Tuple[List[npt.NDArray[np.float64]], List[bool]]:
    """
    Evaluate multiple constraints on a set of player actions and check satisfaction.

    Each constraint function is evaluated with the given actions, and its result
    is checked element-wise against the given epsilon tolerance.

    Parameters
    ----------
    constraints : list of callables
        Each callable should take a list of NumPy arrays (one per player)
        and return a NumPy array of constraint values.
        Constraint satisfaction is defined as all values <= `epsilon`.
    actions : list of numpy.ndarray
        List of 2D NumPy arrays, each representing a player's action vector.
    epsilon : float, optional
        The tolerance for constraint satisfaction (default is 1e-3).

    Returns
    -------
    tuple
        - constraint_values : list of numpy.ndarray
            Raw outputs from each constraint function.
        - constraint_satisfaction : list of bool
            Whether each constraint was satisfied (True) or violated (False).

    Notes
    -----
    Prints a message for each violated constraint and also prints
    "All constraints satisfied" at the end.

    Examples
    --------
    >> def c1(actions): return np.array([[0.0]])
    >> def c2(actions): return np.array([[0.002]])
    >> actions = [np.array([[1.0]])]
    >> constraint_check([c1, c2], actions, epsilon=0.001)
    CONSTRAINT VIOLATION: [[0.002]]
    All constraints satisfied
    ([array([[0.]]), array([[0.002]])], [True, False])
    """
    constraint_values = []
    constraint_satisfaction = []
    for c_idx, constraint in enumerate(constraints):
        c = constraint(actions)
        if not np.all(np.ravel(c) <= epsilon):
            print(f"CONSTRAINT VIOLATION: {c_idx}, {c}")
            constraint_values.append(c)
            constraint_satisfaction.append(False)
        else:
            constraint_values.append(c)
            constraint_satisfaction.append(True)
            print("All constraints satisfied")
    return constraint_values, constraint_satisfaction

def compare_solutions(
    computed_solution: List[float],
    paper_solution: List[float],
    action_sizes: List[int],
    objective_functions: List[Callable[[List[npt.NDArray[np.float64]]], npt.NDArray[np.float64]]],
    solution_name: List[str] = ['Computed', 'Paper']
) -> npt.NDArray[np.float64]:
    """
    Compare two solutions by evaluating their objective function values.

    This function reshapes two flat solution vectors (e.g., from a computed
    algorithm and from a published paper) into per-player action vectors,
    evaluates the provided objective functions for both, prints a table
    comparison, and returns the difference in objective values.

    Parameters
    ----------
    computed_solution : list of float
        Flat list of action values for the computed solution. The length must
        equal the sum of all entries in ``action_sizes``.
    paper_solution : list of float
        Flat list of action values for the paper's reference solution.
    action_sizes : list of int
        Length of each player's action vector. Used to split the flat solutions
        into per-player arrays.
    objective_functions : list of callables
        Functions that take a list of 2D NumPy arrays (one per player) and return
        a NumPy array of objective values.
    solution_name : list of str, optional
        Labels for the two compared solutions, in order. Default is
        ``['Computed', 'Paper']``.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (total_objectives, 1) containing the element-wise
        difference between the computed solution's objective values and the paper
        solution's objective values.

    Notes
    -----
    This function calls:
      - ``construct_vectors`` to split the flat action lists into per-player arrays.
      - ``objective_check`` to evaluate objectives.
      - ``print_table`` to display results.

    Examples
    --------
    >> def obj_sum(players): return np.array([[sum(p.sum() for p in players)]])
    >> computed_solution = [1.0, 2.0]
    >> paper_solution = [1.5, 2.5]
    >> action_sizes = [1, 1]
    >> compare_solutions(computed_solution, paper_solution, action_sizes, [obj_sum])
    Objective Functions
    # Table output here...
    array([[-1.]])
    """
    computed_res = np.array(computed_solution).reshape(-1, 1)
    paper_res = np.array(paper_solution).reshape(-1, 1)

    computed_res_vectors = construct_vectors(computed_res, action_sizes)
    paper_res_vectors = construct_vectors(paper_res, action_sizes)

    computed_res_obj_func = objective_check(objective_functions, computed_res_vectors)
    paper_res_obj_func = objective_check(objective_functions, paper_res_vectors)

    difference = np.array(computed_res_obj_func) - np.array(paper_res_obj_func)
    print("Objective Functions")
    print_table(computed_res_obj_func, paper_res_obj_func, solution_name[0], solution_name[1])
    return difference.reshape(-1, 1)

def one_hot_encoding(
    c_i: List[Union[List[int], List[None]]],
    sizes: List[int],
    num_functions: int
) -> np.ndarray:
    """
    Build a binary matrix indicating which functions use which variables.

    The resulting matrix has shape ``(sum(sizes), num_functions)``, where
    each row corresponds to a variable and each column corresponds to a
    function. An entry ``M[row, col] = 1`` means that function `col` uses
    the variable at `row`.

    Parameters
    ----------
    c_i : list of lists
        Each element corresponds to a variable set (e.g., belonging to one player).
        - If the variable set is used by certain functions, the element is a list of
          integer function indices (0-based).
        - If the variable set is unused, the element should be ``[None]``.
    sizes : list of int
        The number of variables in each variable set. Must have the same length as ``c_i``.
    num_functions : int
        Total number of functions (columns in the resulting matrix).

    Returns
    -------
    numpy.ndarray of shape (sum(sizes), num_functions)
        A binary matrix where rows correspond to variables and columns to functions.

    Raises
    ------
    AssertionError
        If ``len(c_i) != len(sizes)``.

    Examples
    --------
    >> c_i = [[0, 2], [1], [None]]
    >> sizes = [2, 1, 3]
    >> num_functions = 4
    >> one_hot_encoding(c_i, sizes, num_functions)
    array([[1, 0, 1, 0],
           [1, 0, 1, 0],
           [0, 1, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]])
    """
    assert len(c_i) == len(sizes), "c_i and sizes must match in length"

    total_vars = sum(sizes)
    M = np.zeros((total_vars, num_functions), dtype=int)

    # Row offsets per variable set
    offsets = np.cumsum([0] + sizes[:-1])

    for var_idx, funcs in enumerate(c_i):
        # Treat [None] as "uses no functions"
        if funcs == [None]:
            continue
        start = offsets[var_idx]
        end = start + sizes[var_idx]
        M[start:end, funcs] = 1

    return M