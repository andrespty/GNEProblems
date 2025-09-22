import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt


class B1:
    @staticmethod
    def paper_solution():
        return []

    @staticmethod
    def define_players():
        player_vector_sizes = [5,5]
        player_objective_functions = [0, 1]
        player_constraints = [[0, 1, 4, 5, 6], [2, 3, 4, 5, 6]]
        bounds = [(0, 100), (0, 100), (0, 100), (0, 50), (0, 50), (0, 50), (0, 100), (0, 100), (0, 100), (0, 50),
                  (0, 50), (0, 50), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
        bounds_training = [(0, 100), (0, 100), (0, 100), (0, 50), (0, 50), (0, 50), (0, 100), (0, 100), (0, 100),
                           (0, 50), (0, 50), (0, 50), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
        return [player_vector_sizes, player_objective_functions, player_constraints, bounds, bounds_training]

    @staticmethod
    def objective_functions():
        return [B1.obj_func_1, B1.obj_func_2]

    @staticmethod
    def objective_function_derivatives():
        return [B1.obj_func_der_1, B1.obj_func_der_2]

    @staticmethod
    def constraints():
        return [B1.g0, B1.g1, B1.g2, B1.g3, B1.g4, B1.g5, B1.g6]

    @staticmethod
    def constraint_derivatives():
        return [B1.g0_der, B1.g1_der, B1.g2_der, B1.g3_der, B1.g4_der, B1.g5_der, B1.g6_der]

    # @staticmethod
    def abbreviations(x: npt.NDArray[np.float64]) -> float:
        x1 = x[0]
        x2 = x[1]
        S1 = 40 - (40 / 500) * (x1[0] + x1[3] + x2[0] + x2[3])
        S2 = 35 - (35 / 400) * (x1[1] + x1[4] + x2[1] + x2[4])
        S3 = 32 - (32 / 600) * (x1[3] + x1[5] + x2[2] + x2[5])

    @staticmethod
    def obj_func_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        S1 = 40 - 40 / 500 * (x1[0] + x1[3] + x2[0] + x2[3])
        S2 = 35 - 35 / 400 * (x1[1] + x1[4] + x2[1] + x2[4])
        S3 = 32 - 32 / 600 * (x1[2] + x1[5] + x2[2] + x2[5])
        return (15 - S1)(x1[0] + x1[3]) + (15 - S2)(x1[1] + x1[4]) + (15 - S3)(x1[2] + x1[5])

    @staticmethod
    def obj_func_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        S1 = 40 - 40 / 500 * (x1[0] + x1[3] + x2[0] + x2[3])
        S2 = 35 - 35 / 400 * (x1[1] + x1[4] + x2[1] + x2[4])
        S3 = 32 - 32 / 600 * (x1[2] + x1[5] + x2[2] + x2[5])
        return (15 - S1)(x2[0] + x2[3]) + (15 - S2)(x2[1] + x2[4]) + (15 - S3)(x2[2] + x2[3])

    @staticmethod
    def obj_func_der_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        return np.array([15 - 40 - 80 / 500 * x1[0] - 80 / 500 * x1[3] - 40 / 500 * x2[0] - 40 / 500 * x2[3],
                         15 - 35 - 70 / 400 * x1[1] - 70 / 400 * x1[4] - 35 / 400 * x2[1] - 35 / 400 * x2[4],
                         15 - 32 - 64 / 600 * x1[2] - 64 / 600 * x1[5] - 32 / 600 * x2[2] - 32 / 600 * x2[5],

                         15 - 40 - 80 / 500 * x1[3] - 80 / 500 * x1[0] - 40 / 500 * x2[0] - 40 / 500 * x2[3],
                         15 - 35 - 70 / 400 * x1[4] - 70 / 400 * x1[1] - 35 / 400 * x2[1] - 35 / 400 * x2[4],
                         15 - 32 - 64 / 600 * x1[5] - 64 / 600 * x1[2] - 32 / 600 * x2[2] - 32 / 600 * x2[5]]).reshape(
            -1, 1)

    @staticmethod
    def obj_func_der_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        return np.array([15 - 40 - 80 / 500 * x2[0] - 80 / 500 * x2[3] - 40 / 500 * x1[0] - 40 / 500 * x1[3],
                         15 - 35 - 70 / 400 * x2[1] - 70 / 400 * x2[4] - 35 / 400 * x1[1] - 35 / 400 * x1[4],
                         15 - 32 - 64 / 600 * x2[2] - 64 / 600 * x2[5] - 32 / 600 * x1[2] - 32 / 600 * x1[5],

                         15 - 40 - 80 / 500 * x2[3] - 80 / 500 * x2[0] - 40 / 500 * x1[0] - 40 / 500 * x1[3],
                         15 - 35 - 70 / 400 * x2[4] - 70 / 400 * x2[1] - 35 / 400 * x1[1] - 35 / 400 * x1[4],
                         15 - 32 - 64 / 600 * x2[5] - 64 / 600 * x2[2] - 32 / 600 * x1[2] - 32 / 600 * x1[5]]).reshape(
            -1, 1)

    @staticmethod
    def g0(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        return x1[0] + x1[1] + x1[2] - 100

    @staticmethod
    def g1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        return x1[3] + x1[4] + x1[5] - 50

    @staticmethod
    def g2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        return x2[0] + x2[1] + x2[2] - 100

    @staticmethod
    def g3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        return x2[3] + x2[4] + x2[5] - 50

    @staticmethod
    def g4(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        S1 = 40 - 40 / 500 * (x1[0] + x1[3] + x2[0] + x2[3])
        S2 = 35 - 35 / 400 * (x1[1] + x1[4] + x2[1] + x2[4])
        S3 = 32 - 32 / 600 * (x1[2] + x1[5] + x2[2] + x2[5])
        return abs(S1 - S2) - 1

    @staticmethod
    def g5(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        S1 = 40 - 40 / 500 * (x1[0] + x1[3] + x2[0] + x2[3])
        S2 = 35 - 35 / 400 * (x1[1] + x1[4] + x2[1] + x2[4])
        S3 = 32 - 32 / 600 * (x1[2] + x1[5] + x2[2] + x2[5])
        return abs(S1 - S3) - 1

    @staticmethod
    def g6(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        S1 = 40 - 40 / 500 * (x1[0] + x1[3] + x2[0] + x2[3])
        S2 = 35 - 35 / 400 * (x1[1] + x1[4] + x2[1] + x2[4])
        S3 = 32 - 32 / 600 * (x1[2] + x1[5] + x2[2] + x2[5])
        return abs(S2 - S3) - 1

    @staticmethod
    def g0_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).reshape(-1, 1)

    @staticmethod
    def g1_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]]).reshape(-1, 1)

    @staticmethod
    def g2_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]]).reshape(-1, 1)

    @staticmethod
    def g3_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]).reshape(-1, 1)

    @staticmethod
    def g4_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([[40 / 500, 35 / 400, 0, 40 / 500, 35 / 400, 0,
                          40 / 500, 35 / 400, 0, 40 / 500, 35 / 400, 0]]).reshape(-1, 1)

    @staticmethod
    def g5_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([[40 / 500, 0, 32 / 600, 40 / 500, 0, 32 / 600,
                          40 / 500, 0, 32 / 600, 40 / 500, 0, 32 / 600]]).reshape(-1, 1)

    @staticmethod
    def g6_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([[0, 35 / 400, 32 / 600, 0, 35 / 400, 32 / 600,
                          0, 35 / 400, 32 / 600, 0, 35 / 400, 32 / 600]]).reshape(-1, 1)


