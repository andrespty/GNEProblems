import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt

class A1U:
    @staticmethod
    def paper_solution():
        value_1 = [0.29923815223336,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805]
        return [value_1]

    @staticmethod
    def define_players():
        B = 1
        player_vector_sizes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        player_objective_functions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # change to all 0s
        player_constraints = [[1,2], [0,3], [0,3], [0,3], [0,3], [0,3], [0,3], [0,3], [0,3], [0,3]]
        return [player_vector_sizes, player_objective_functions, player_constraints]

    @staticmethod
    def objective_functions():
        return [A1U.obj_func]

    @staticmethod
    def objective_function_derivatives():
        return [A1U.obj_func_der]

    @staticmethod
    def constraints():
        return [A1U.g0, A1U.g1, A1U.g2, A1U.g3]

    @staticmethod
    def constraint_derivatives():
        return [A1U.g0_der, A1U.g1_der, A1U.g2_der, A1U.g3_der]

    @staticmethod
    def obj_func(x):
        # x: numpy array (N, 1)
        # B: constant
        x = np.concatenate(x)
        S = np.sum(x)
        B=1
        obj = (-x / S) * (1 - S / B)
        return obj

    @staticmethod
    def obj_func_der(x):
        # x: numpy array (N,1)
        # B: constant
        x = np.concatenate(x).reshape(-1, 1)
        B=1
        S = sum(x)
        # print(S)
        obj = ((x - S) / S ** 2) + (1 / B)
        return obj

    # === Constraint Functions ===
    @staticmethod
    def g0(x):
        # x: numpy array (N,1)
        # B: constant
        B=1
        return sum(x) - B

    @staticmethod
    def g1(x):
        # lower bound
        return 0.3 - x[0]

    @staticmethod
    def g2(x):
        #  upper bound
        return x[0] - 0.5

    @staticmethod
    def g3(x):
        t = np.vstack([s.reshape(-1,1) for s in x[1:]])
        return 0.01 - t

    @staticmethod
    def g0_der(x):
        return 1

    @staticmethod
    def g1_der(x):
        return -1

    @staticmethod
    def g2_der(x):
        return 1

    @staticmethod
    def g3_der(x):
        return -1