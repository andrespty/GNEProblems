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

def get_problem(problem_n):
    # Define the problem
    obj = problem_n.objective_functions()
    obj_der = problem_n.objective_function_derivatives()
    c = problem_n.constraints()
    c_der = problem_n.constraint_derivatives()

    # Describe Players responsibilities
    p = problem_n.define_players()
    return [obj,obj_der,c,c_der,p]

if __name__ == '__main__':
    problem = A8

    (player_vector_sizes,
     player_objective_functions,
     player_constraints,
     bounds,
     bounds_training) = get_problem(problem)[4]

    # Define the problem solver
    solver1 = GeneralizedNashEquilibriumSolver(
        *get_problem(problem)[:4],
        player_objective_functions,
        player_constraints,
        bounds,
        player_vector_sizes,
        useBounds=False
    )

    # Set initial point for solution
    x1 = np.array([[0]], dtype=np.float64)
    x2 = np.array([[0]], dtype=np.float64)
    x3 = np.array([[0]], dtype=np.float64)
    actions = [x1, x2, x3]
    dual = [0, 0]

    # Solve problem
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




