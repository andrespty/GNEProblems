import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt


class A17:
    @staticmethod
    def paper_solution():
        value_1 = [0.00000737,
                   11.00002440,
                   7.99997560]
        return [value_1]

    @staticmethod
    def define_players():
        player_vector_sizes = [1, 1, 1]
        player_objective_functions = [0, 0, 1]
        player_constraints = [[0, 1], [0, 1], [0, 1]]
        bounds = [(0.0, 100), (0.0, 100), (0.0, 100), (0.0, 100), (0.0, 100)]
        bounds_training = [(0.0, 100), (0.0, 100), (0.0, 100), (0.0, 100), (0.0, 100)]
        return [player_vector_sizes, player_objective_functions, player_constraints, bounds, bounds_training]

    @staticmethod
    def objective_functions():
        return [A17.obj_func_1, A17.obj_func_2]

    @staticmethod
    def objective_function_derivatives():
        return [A17.obj_func_der_1, A17.obj_func_der_2]

    @staticmethod
    def constraints():
        return [A17.g0, A17.g1]

    @staticmethod
    def constraint_derivatives():
        return [A17.g0_der, A17.g1_der]

    @staticmethod
    def obj_func_1(x: npt.NDArray[np.float64]):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return x1 ** 2 + x1 * x2 + x2 ** 2 + x1 * x3 + x2 * x3 - 25 * x1 - 38 * x2

    @staticmethod
    def obj_func_2(x: npt.NDArray[np.float64]):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return x3 ** 2 + x1 * x3 + x2 * x3 - 25 * x3

    @staticmethod
    def obj_func_der_1(x: npt.NDArray[np.float64]):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return 2 * x1 + x2 + x3 - 25

    @staticmethod
    def obj_func_der_2(x: npt.NDArray[np.float64]):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return x1 + 2 * x2 + x3 - 38

    @staticmethod
    def obj_func_der_3(x: npt.NDArray[np.float64]):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return 2 * x3 + x1 + x2 - 25

    @staticmethod
    def g0(x: npt.NDArray[np.float64]):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return x1 + 2 * x2 - x3 - 14

    @staticmethod
    def g1(x: npt.NDArray[np.float64]):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return 3 * x1 + 2 * x2 + x3 - 30

    @staticmethod
    def g0_der(x: npt.NDArray[np.float64]):
        return np.array([[1, 2, -1]]).reshape(-1, 1)

    @staticmethod
    def g1_der(x: npt.NDArray[np.float64]):
        return np.array([[3, 2, 1]]).reshape(-1,1)
