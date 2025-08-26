import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt


class A14U:
    @staticmethod
    def paper_solution():
        value_1 = [ 0.08999991899425, 0.08999991899426, 0.08999991899425,
                    0.08999991899425, 0.08999991899425, 0.08999991899425,
                    0.08999991899425, 0.08999991899426, 0.08999991899425,
                    0.08999991899425]
        return [value_1]

    @staticmethod
    def define_players():
        n = 10
        player_vector_sizes = [1 for _ in range(n)]
        player_objective_functions = [0 for _ in range(n)]
        player_constraints = [[0,1] for _ in range(n)]
        return [player_vector_sizes, player_objective_functions, player_constraints]

    @staticmethod
    def objective_functions():
        return [A14U.obj_func]

    @staticmethod
    def objective_function_derivatives():
        return [A14U.obj_func_der]

    @staticmethod
    def constraints():
        return [A14U.g0, A14U.g1]

    @staticmethod
    def constraint_derivatives() -> List:
        return [A14U.g0_der, A14U.g1_der]

    @staticmethod
    def obj_func(x):
        """
        Parameters
        ----------
        x : list of numpy.ndarray shape (any, 1)
            A list of NumPy arrays to be concatenated along their first axis.

        Returns
        -------
        numpy.ndarray shape (any, 1)
        """
        x = np.concatenate(x).reshape(-1, 1)
        s = np.sum(x)
        b = 1
        obj = (-x / s) * (1 - s / b)
        return obj

    @staticmethod
    def obj_func_der(x):
        """
        Parameters
        ----------
        x : list of numpy.ndarray shape (any, 1)
            A list of NumPy arrays to be concatenated along their first axis.

        Returns
        -------
        numpy.ndarray shape (any, 1)
        """
        x = np.concatenate(x).reshape(-1, 1)
        b = 1
        s = sum(x)
        obj = ((x - s) / s ** 2) + (1 / b)
        return obj

    @staticmethod
    def g0(x):
        B = 1
        return sum(x) - B

    @staticmethod
    def g1(x):
        # lower bound
        x = np.concatenate(x).reshape(-1, 1)
        return 0.01 - x
    
    @staticmethod
    def g0_der(x):
        return 1
    
    @staticmethod
    def g1_der(x):
        return -1