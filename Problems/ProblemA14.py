import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt


class A14:
    @staticmethod
    def paper_solution():
        value_1 = [ 0.08999991899425, 0.08999991899426, 0.08999991899425,
                    0.08999991899425, 0.08999991899425, 0.08999991899425,
                    0.08999991899425, 0.08999991899426, 0.08999991899425,
                    0.08999991899425]
        return [value_1]

    @staticmethod
    def define_players():
        B = 1
        player_vector_sizes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        player_objective_functions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        player_constraints = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        bounds = [(0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B),
                (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0, 10)]
        bounds_training = [(0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B),
                            (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0, 10)]
        return [player_vector_sizes, player_objective_functions,
        player_constraints, bounds, bounds_training]

    @staticmethod
    def objective_functions():
        return [A14.obj_func]

    @staticmethod
    def objective_function_derivatives():
        return [A14.obj_func_der]

    @staticmethod
    def constraints():
        return [A14.g0]

    @staticmethod
    def constraint_derivatives() -> List:
        return [A14.g0_der]

    @staticmethod
    def obj_func(x: npt.NDArray) -> float:
        W = sum(x)
        B = 1
        if W <= 0:
            return 0.0
            return float((x[0] / W) * (1 - W/B))

    @staticmethod
    def obj_func_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        W = sum(x)
        if W <= 0:
            return np.zeros_like(x)
            w_i = x[0]
            grad = (W - w_i) / W**2 - 1
            return grad

    @staticmethod
    def g0(x):
        B = 1
        return x.sum() - B

    @staticmethod
    def g0_der(x):
        return 1