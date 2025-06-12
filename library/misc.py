import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt


def flatten_variables(vectors, scalars):
    """
    Flattens a list of vectors and scalars into a single list for optimization.
    """
    return np.hstack([v.flatten() for v in vectors] + [scalars])


def construct_vectors(actions: npt.NDArray[np.float64], action_sizes: List[int]) -> List[npt.NDArray[np.float64]]:
    """
    Input:
      actions: np.array of all players' actions. Shape (sum(all actions), 1)
      action_sizes: list of sizes of each player's action vector
    Output:
      python list of 2d np.arrays
    """
    value_array = np.array(actions)
    indices = np.cumsum(action_sizes)
    return np.split(value_array, indices[:-1])


def deconstruct_vectors(vectors: List[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    """
    Input:
        vectors: list of 2d np.arrays
    Output:
        np.array of all players' actions in a vector. Shape (sum(all actions), 1)
    """
    return np.concatenate(vectors)


def create_wrapped_function(
        original_func: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        vars: List[npt.NDArray[np.float64]],
        player_idx: int
):
    player_var = vars[player_idx]
    fixed_vars = vars[:player_idx] + vars[player_idx + 1:]  # list of np vectors

    def wrap_func(player_var_opt):
        player_var_opt = np.array(player_var_opt).reshape(-1, 1)
        new_vars = fixed_vars[:player_idx] + [player_var_opt] + fixed_vars[player_idx:]
        return original_func(new_vars)

    return wrap_func


def objective_check(objective_functions: List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
                    actions: List[npt.NDArray[np.float64]]):
    objective_values = []
    for objective in objective_functions:
        o = objective(actions)
        objective_values.append(o)
    return objective_values


def constraint_check(constraints: List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
                     actions: List[npt.NDArray[np.float64]], epsilon=1e-3):
    for constraint in constraints:
        c = constraint(actions)
        if not np.all(np.ravel(c) <= epsilon):
            print(f"CONSTRAINT VIOLATION: {c}")
            return False
    print("All constraints satisfied")
    return True


def print_table(vec1, vec2, header1="Vector 1", header2="Vector 2"):
    print(f"{header1:^10} | {header2:^10}")  # Header
    print("-" * 23)
    for v1, v2 in zip(vec1, vec2):
        # Extract scalar value from NumPy arrays
        v1_scalar = v1.item() if isinstance(v1, np.ndarray) else v1
        v2_scalar = v2.item() if isinstance(v2, np.ndarray) else v2
        print(f"{v1_scalar:^10.4f} | {v2_scalar:^10.4f}")  # Align values with 4 decimal places


def compare_solutions(
        computed_solution: List[float],
        paper_solution: List[float],
        action_sizes: List[int],
        objective_functions: List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
        solution_name=['Computed', 'Paper']
):
    computed_NE = np.array(computed_solution).reshape(-1, 1)
    paper_NE = np.array(paper_solution).reshape(-1, 1)

    computed_NE_vectors = construct_vectors(computed_NE, action_sizes)
    paper_NE_vectors = construct_vectors(paper_NE, action_sizes)

    computed_NE_obj_func = objective_check(objective_functions, computed_NE_vectors)
    paper_NE_obj_func = objective_check(objective_functions, paper_NE_vectors)

    difference = np.array(computed_NE_obj_func) - np.array(paper_NE_obj_func)
    #     print(f"Average difference between {solution_name[0]} and {solution_name[1]}: ", np.mean(difference))
    print("Objective Functions")
    print_table(computed_NE_obj_func, paper_NE_obj_func, solution_name[0], solution_name[1])
    return difference.reshape(-1, 1)


def check_nash_equillibrium(
        result: List[np.float64],
        action_sizes: List[int],
        player_objective_function: List[int],
        objective_functions: List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
        constraints: List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
        player_constraints: List[List[int]],
        bounds: List[Tuple[float, float]],
        paper_res: List[float] = None,
        epsilon=1e-3
):
    computed_NE = np.array(result).reshape(-1, 1)
    computed_NE_vectors = construct_vectors(computed_NE, action_sizes)
    print('Computed Solution: \n', computed_NE_vectors)
    action_splits = np.cumsum(np.insert(action_sizes, 0, 0))

    # Get the obj function values at the current NE
    computed_NE_obj_func = objective_check(objective_functions, computed_NE_vectors)
    print('Computed Objective Function: \n', np.array(computed_NE_obj_func).reshape(-1, 1))

    # Check that constraints are being satisfied
    if not constraint_check(constraints, computed_NE_vectors):
        return

    if paper_res is not None:
        compare_solutions(
            computed_NE,
            paper_res,
            action_sizes,
            objective_functions,
        )
    print('--------------------------------------')
    # Optimize each player by fixing the opponents
    for player_idx, p_o_idx in enumerate(player_objective_function):
        p_var = computed_NE_vectors[player_idx]
        p_objective = objective_functions[p_o_idx]  # this is a function
        p_constraints = [constraints[c_idx] for c_idx in player_constraints[p_o_idx] if c_idx]

        wrapped_p_objective = create_wrapped_function(p_objective, computed_NE_vectors, player_idx)

        optimization_constraints = [
            {'type': 'ineq', 'fun': lambda x: create_wrapped_function(constraint, computed_NE_vectors, player_idx)(x)}
            for constraint in p_constraints
        ]

        p_var_0 = np.zeros_like(p_var).flatten()
        # p_var_0 = p_var.flatten()
        player_bounds = bounds[action_splits[player_idx]:action_splits[player_idx + 1]]
        result = minimize(
            wrapped_p_objective,
            p_var_0,
            method='SLSQP',
            bounds=player_bounds,
            constraints=optimization_constraints,
            options={
                'disp': False,
                'maxiter': 1000,
                'ftol': 1e-5
            }
        )
        # Report
        fixed_vars = computed_NE_vectors[:player_idx] + computed_NE_vectors[player_idx + 1:]
        opt_actions = np.array(result.x).reshape(-1, 1)
        new_vars = fixed_vars[:player_idx] + [opt_actions] + fixed_vars[player_idx:]
        opt_obj_func = objective_check(objective_functions, new_vars)
        print(f"Player {player_idx + 1}")
        print(f"Optimized Actions: {result.x}\nOptimized Function value: {result.fun}")
        print('Optimized Objective Functions: \n', np.array(opt_obj_func).reshape(-1, 1))
        constraint_check(constraints, new_vars)
        difference = compare_solutions(
            computed_NE,
            deconstruct_vectors(new_vars),
            action_sizes,
            objective_functions,
            solution_name=['Computed', 'Optimized']
        )
        print(f"Difference between Objective Functions of Player {player_idx + 1}: {difference[p_o_idx]}")
        if difference[p_o_idx] > epsilon:
            print('Not at the NE')
        else:
            print(f"Computed Solution is at the NE for Player {player_idx + 1}")
        print('--------------------------------------')
    return

def repeat_items(items, sizes):
    """
    Input:
      items: list of items     [1,2,3,4,5]
      sizes: list of sizes     [2,3,1,4,1]
    Output:
      list of repeated items   [1,1,2,2,2,3,4,4,4,4,5]
    """
    return np.array(np.repeat(items, sizes, axis=0).tolist())

def vectorized_sigmoid(x: npt.NDArray[np.float64], bounds: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Input:
      x: 2d np.array (x,1)
      bounds: 2d np.array (x, 1)
    Output:
      2d np.array (x,1)
    """
    x = np.clip(x, -500, 500)
    lb = bounds[:, 0].reshape(-1,1)
    ub = bounds[:, 1].reshape(-1,1)
    sigmoid = 1 / (1 + np.exp(-x))
    return sigmoid * (ub - lb) + lb