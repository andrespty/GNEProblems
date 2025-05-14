import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt
from GNESolver5 import *
from ProblemFunctions import *

def test_solver():

    print('Testing Solver')
    objective_functions = [A8.obj_func_1, A8.obj_func_2, A8.obj_func_3]
    objective_function_derivatives = [A8.obj_func_der_1, A8.obj_func_der_2, A8.obj_func_der_3]
    constraint_functions = [A8.g0, A8.g1]
    constraint_function_derivatives = [A8.g0_der, A8.g1_der]

    player_vector_sizes = [1, 1, 1]
    player_objective_functions = [0, 1, 2]
    player_constraints = [[0, 1], [0, 1], [None]]
    bounds = [(0, 100), (0, 100), (0, 2), (0, 100), (0, 100)]
    bounds_training = [(0, 100), (0, 100), (0, 2), (0, 100), (0, 100)]

    solver1 = GeneralizedNashEquilibriumSolver(
        objective_functions,
        objective_function_derivatives,
        constraint_functions,
        constraint_function_derivatives,
        player_objective_functions,
        player_constraints,
        bounds,
        player_vector_sizes,
        useBounds=False
    )
    x1 = np.array([[0]], dtype=np.float64)
    x2 = np.array([[0]], dtype=np.float64)
    x3 = np.array([[0]], dtype=np.float64)
    actions = [x1, x2, x3]
    dual = [0, 0]

    sol = solver1.solve_game(flatten_variables(actions, dual), bounds_training)
    solver1.summary(A8.value_1)
    sol[0].x
    solver1.NE_check()
    check_nash_equillibrium(
        sol[0].x[:3],
        player_vector_sizes,
        player_objective_functions,
        objective_functions,
        constraint_functions,
        player_constraints,
        bounds,
        A8.value_1
    )
    check_nash_equillibrium(
        A8.value_1,
        player_vector_sizes,
        player_objective_functions,
        objective_functions,
        constraint_functions,
        player_constraints,
        bounds,
        A8.value_2
    )
    return

if __name__ == '__main__':
    test_solver()

