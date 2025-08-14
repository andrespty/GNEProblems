import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt


class A11:
    @staticmethod
    def paper_solution():
        value_1 = [0.75002417863494, 0.2500241786374]
        return [value_1]

    @staticmethod
    def define_players():
        player_vector_sizes = [1, 1]
        player_objective_functions = [0, 1]
        player_constraints = [[0], [0]]
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 100.0)]
        bounds_training = [(0.0, 1.0), (0.0, 1.0), (0.0, 100.0)]
        return [player_vector_sizes, player_objective_functions, player_constraints, bounds, bounds_training]

    @staticmethod
    def objective_functions():
        return [A11.obj_func_1, A11.obj_func_2]

    @staticmethod
    def objective_function_derivatives():
        return [A11.obj_func_der_1, A11.obj_func_der_2]

    @staticmethod
    def constraints():
        return [A11.g0]

    @staticmethod
    def constraint_derivatives():
        return [A11.g0_der]

    @staticmethod
    def obj_func_1(x: npt.NDArray[np.float64]) -> float:
        x1 = x[0]
        return (x1 - 1) ** 2

    @staticmethod
    def obj_func_2(x: npt.NDArray[np.float64]) -> float:
        x2 = x[1]
        return (x2 - 0.5) ** 2

    @staticmethod
    def obj_func_der_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        grad = 2 * (x[0] - 1)  # d/dx1 of (x1 - 1)^2
        return grad

    @staticmethod
    def obj_func_der_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        grad = 2 * (x[1] - 0.5)  # d/dx2 of (x2 - 0.5)^2
        return grad

    @staticmethod
    def g0(x: npt.NDArray[np.float64]) -> float:
        return x[0] + x[1] - 1

    @staticmethod
    def g0_der(x: npt.NDArray[np.float64]) -> float:
        return np.array([[1, 1]]).reshape(-1, 1)