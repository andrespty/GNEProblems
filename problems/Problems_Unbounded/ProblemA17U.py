import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt


class A17U:
    @staticmethod
    def paper_solution():
        value_1 = [0.00000737,
                   11.00002440,
                   7.99997560]
        return [value_1]

    @staticmethod
    def define_players():
        player_vector_sizes = [1, 1, 1]
        player_objective_functions = [0, 1, 2]
        player_constraints = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
        return [player_vector_sizes, player_objective_functions, player_constraints]

    @staticmethod
    def objective_functions():
        return [A17U.obj_func_1,A17U.obj_func_1, A17U.obj_func_2]

    @staticmethod
    def objective_function_derivatives():
        return [A17U.obj_func_der_1, A17U.obj_func_der_2, A17U.obj_func_der_3]

    @staticmethod
    def constraints():
        return [A17U.g0, A17U.g1, A17U.g2]

    @staticmethod
    def constraint_derivatives():
        return [A17U.g0_der, A17U.g1_der, A17U.g2_der]

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
    def g2(x: npt.NDArray[np.float64]):
        x = np.concatenate(x).reshape(-1, 1)
        return 0 - x

    @staticmethod
    def g0_der(x: list[np.ndarray]) -> np.ndarray:
        grad = np.zeros((3, 1))
        grad[0] = 1
        grad[1] = 2
        grad[2] = -1
        return grad

    @staticmethod
    def g1_der(x: list[np.ndarray]) -> np.ndarray:
        grad = np.zeros((3, 1))
        grad[0] = 3
        grad[1] = 2
        grad[2] = 1
        return grad


    @staticmethod
    def g2_der(x: npt.NDArray[np.float64]):
        return -1