import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt


class A12U:
    @staticmethod
    def paper_solution():
        value_1 = [5.33331555561568, 5.33331555561]
        return [value_1]

    @staticmethod
    def define_players():
        player_vector_sizes = [1, 1]
        player_objective_functions = [0, 1]
        player_constraints = [[0, 1], [0, 1]]
        return [player_vector_sizes, player_objective_functions, player_constraints]

    @staticmethod
    def objective_functions():
        return [A12U.obj_func_1, A12U.obj_func_2]

    @staticmethod
    def objective_function_derivatives():
        return [A12U.obj_func_der_1, A12U.obj_func_der_2]

    @staticmethod
    def constraints():
        return [A12U.g0, A12U.g1]

    @staticmethod
    def constraint_derivatives():
        return [A12U.g0_der, A12U.g1_der]

    @staticmethod
    def obj_func_1(x: npt.NDArray[np.float64]) -> float:
        return x[0] * (x[0] + x[1] - 16)

    @staticmethod
    def obj_func_2(x: npt.NDArray[np.float64]) -> float:
        return x[1] * (x[0] + x[1] - 16)

    @staticmethod
    def obj_func_der_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        grad = 2 * x[0] + x[1] - 16
        return grad

    @staticmethod
    def obj_func_der_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        grad = x[0] + 2 * x[1] - 16
        return grad
    
    @staticmethod
    def g0(x):
        return -10 - np.concatenate(x).reshape(-1, 1)
    
    @staticmethod
    def g1(x):
        return np.concatenate(x).reshape(-1,1) - 10
    
    @staticmethod
    def g0_der(x):
        return -1
    
    @staticmethod
    def g1_der(x):
        return 1
