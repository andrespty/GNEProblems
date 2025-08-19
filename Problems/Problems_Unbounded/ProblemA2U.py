import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt


class A2U:
    @staticmethod
    def paper_solution():
        value_1 = [0.29962894677774, 0.00997828224734, 0.00997828224734,
                   0.00997828224734, 0.59852469355630, 0.02187270661760,
                   0.00999093169361, 0.00999093169361, 0.00999093169361,
                   0.00999093169361]

        value_2 = [0.29962898846513, 0.00997828313762, 0.00997828313762,
                   0.00997828313762, 0.59745624992082, 0.02220301920403,
                   0.01013441012117, 0.01013441012117, 0.01013441012117,
                   0.01013441012117]

        return [value_1, value_2]

    @staticmethod
    def define_players():
        B = 1
        player_vector_sizes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        player_objective_functions = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        player_constraints = [[2, 3], [0, 4], [0, 4], [0, 4], [0, 1, 4], [0, 1, 4], [0, 4], [0, 4], [0, 4, 5], [0, 4, 6]]
        return [player_vector_sizes, player_objective_functions, player_constraints]

    @staticmethod
    def objective_functions():
        return [A2U.obj_func_1, A2U.obj_func_2]

    @staticmethod
    def objective_function_derivatives():
        return [A2U.obj_func_der_1, A2U.obj_func_der_2]

    @staticmethod
    def constraints():
        return [A2U.g0, A2U.g1, A2U.g2, A2U.g3, A2U.g4, A2U.g5, A2U.g6]

    @staticmethod
    def constraint_derivatives():
        return [A2U.g0_der, A2U.g1_der, A2U.g2_der, A2U.g3_der, A2U.g4_der, A2U.g5_der, A2U.g6_der]

    @staticmethod
    def obj_func_1(x):
        # x: numpy array (N, 1)
        x = np.concatenate(x).reshape(-1, 1)
        B = 1
        S = np.sum(x)
        obj = (-x / S) * (1 - S / B)
        return obj

    @staticmethod
    def obj_func_2(x):
        # x: numpy array (N,1)
        # B: constant
        x = np.concatenate(x).reshape(-1, 1)
        B = 1
        S = np.sum(x)
        obj = (-x / S) * (1 - S / B) ** 2
        return obj

    @staticmethod
    def obj_func_der_1(x):
        # x: numpy array (N,1)
        B = 1
        x = np.concatenate(x).reshape(-1, 1)
        S = sum(x)
        obj = (x - S) / S ** 2 + 1 / B
        return obj

    @staticmethod
    def obj_func_der_2(x):
        # x: numpy array (N,1)
        B = 1
        x = np.concatenate(x).reshape(-1, 1)
        S = sum(x) + 1e-3
        obj = (2 * B * (S ** 2) - (B ** 2) * S - S ** 3 - x * (S ** 2) + x * (B ** 2)) / (S ** 2)
        return obj

    @staticmethod
    def g0(x):
        # x: numpy array (N,1)
        B = 1
        return sum(x) - B

    @staticmethod
    def g1(x):
        # x: numpy array (N,1)
        B = 1
        return 0.99 - sum(x)

    @staticmethod
    def g2(x):
        return 0.3 - x[0]

    @staticmethod
    def g3(x):
        return x[0] - 0.5

    @staticmethod
    def g4(x):
        t = np.vstack([s.reshape(-1, 1) for s in x[1:]])
        return 0.01 - t

    @staticmethod
    def g5(x):
        return x[8] - 0.06

    @staticmethod
    def g6(x):
        return x[9] - 0.05

    @staticmethod
    def g0_der(x):
        return 1

    @staticmethod
    def g1_der(x):
        return -1

    @staticmethod
    def g2_der(x):
        return -1

    @staticmethod
    def g3_der(x):
        return 1

    @staticmethod
    def g4_der(x):
        return -1

    @staticmethod
    def g5_der(x):
        return 1

    @staticmethod
    def g6_der(x):
        return 1