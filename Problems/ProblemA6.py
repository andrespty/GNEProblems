import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt

class A6:
    @staticmethod
    def paper_solution():
        value_1 = [0.99987722673822, 2.31570964703584, 0.99989251930167, 1.31499923583926, 0.99989852480755,
                   0.99992298465841, 1.09709158271764]
        value_2 = [0.99973555394222, 2.31634992067271, 0.99976846015730, 1.31481981480565, 0.99993110204166,
                   0.99983409362034, 1.09703474801283]
        value_3 = [0.99987722673822, 2.31570964703584, 0.99989251930167, 1.31499923583926, 0.99989852480755,
                   0.99992298465841, 1.09709158271764]
        return [value_1, value_2, value_3]

    @staticmethod
    def define_players():
        player_vector_sizes = [1, 1, 1]
        player_objective_functions = [0, 1, 2]
        player_constraints = [[0, 1], [0, 1], [None]]
        bounds = [(0, 100), (0, 100), (0, 2), (0, 100), (0, 100)]
        bounds_training = [(0, 100), (0, 100), (0, 2), (0, 100), (0, 100)]
        return [player_vector_sizes, player_objective_functions, player_constraints, bounds, bounds_training]

    @staticmethod
    def objective_functions():
        return [A6.obj_func_1, A6.obj_func_2, A6.obj_func_3]

    @staticmethod
    def objective_function_derivatives():
        return [A6.obj_func_der_1, A6.obj_func_der_2, A6.obj_func_der_3]

    @staticmethod
    def constraints():
        return [A6.g0, A6.g1]

    @staticmethod
    def constraint_derivatives():
        return [A6.g0_der, A6.g1_der]

    @staticmethod
    def obj_func_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
      x1 = x[0]
      x2 = x[1]
      x3 = x[2]
      return -x1

    @staticmethod
    def obj_func_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
      x1 = x[0]
      x2 = x[1]
      x3 = x[2]
      return (x2-0.5)**2

    @staticmethod
    def obj_func_3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
      x1 = x[0]
      x2 = x[1]
      x3 = x[2]
      return (x3 - 1.5*x1)**2

    @staticmethod
    def obj_func_der_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return -1 + 0 * x1

    @staticmethod
    def obj_func_der_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return 2 * x2 - 1

    @staticmethod
    def obj_func_der_3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return 2 * x3 - 3 * x1

    @staticmethod
    def g0(x):
        x1, x2, x3 = x
        return (x1 + x2 - 1)[0]

    @staticmethod
    def g1(x):
        x1, x2, x3 = x
        return (x3 - x1 - x2)[0]

    @staticmethod
    def g0_der(x1):
        return np.array([[1, 1, 0]]).reshape(-1, 1)

    @staticmethod
    def g1_der(x1):
        return np.array([[-1, -1, 0]]).reshape(-1, 1)