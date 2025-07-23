import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt

class A10:
    K=8
    N=7

    @staticmethod
    def define_players():
        player_vector_sizes = [A10.K for _ in range(A10.N)]
        player_objective_functions = [0 for _ in range(A10.N)]  # change to all 0s
        player_constraints = [[0] for _ in range(A10.N)]
        return [player_vector_sizes, player_objective_functions, player_constraints]

    @staticmethod
    def objective_functions():
        return [A10.obj_func]

    @staticmethod
    def objective_function_derivatives():
        return [A10.obj_func_der]

    @staticmethod
    def constraints():
        return [A10.g0]

    @staticmethod
    def constraint_derivatives():
        return [A10.g0_der]

    @staticmethod
    def obj_func(x):
        # x: numpy array (N, 1)
        # B: constant
        return sum(x)

    @staticmethod
    def obj_func_der(x):
        return 1

