import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt

class A18:
    @staticmethod
    def paper_solution():
        return []

    @staticmethod
    def define_players():
        player_vector_sizes = [6, 6]
        player_objective_functions = [0, 1]
        player_constraints = [[0,1,4,5,6],[2,3,4,5,6]]
        bounds = [(0.0, 100), (0.0, 100), (0.0, 100), (0.0, 50), (0.0, 50), (0.0, 50), (0.0, 100), (0.0, 100), (0.0, 100), (0.0, 50), (0.0, 50), (0.0, 50), (0.0, 1) , (0.0, 1), (0.0, 1), (0.0, 1), (0.0,1), (0.0, 1), (0.0, 1)]
        bounds_training = [(0.0, 100), (0.0, 100), (0.0, 100), (0.0, 50), (0.0, 50), (0.0, 50), (0.0, 100), (0.0, 100), (0.0, 100), (0.0, 50), (0.0, 50), (0.0, 50), (0.0, 1) , (0.0, 1), (0.0, 1), (0.0, 1), (0.0,1), (0.0, 1), (0.0, 1)]
        return [player_vector_sizes, player_objective_functions, player_constraints, bounds, bounds_training]
     
    @staticmethod
    def objective_functions():
        return [A18.obj_func_1, A18.obj_func_2]

    @staticmethod
    def objective_function_derivatives():
        return [A18.obj_func_der_1, A18.obj_func_der_2]
 
    @staticmethod
    def constraints():
        return [A18.g0, A18.g1, A18.g2, A18.g3, A18.g4, A18.g5, A18.g6]

    @staticmethod
    def constraint_derivatives():
        return [A18.g0_der, A18.g1_der, A18.g2_der, A18.g3_der, A18.g4_der, A18.g5_der, A18.g6_der]
   
    #@staticmethod
    def abbreviations(x: npt.NDArray[np.float64]) -> float:
        S1 = 40 - (40/500)* (x[0] + x[5] + x[8] + x[11])
        S2 = 35 - 0.07* (x[3] + x[6] + x[9] + x[12])
        S3 = 32 - 0.0533* (x[4] + x[7] + x[10] + x[13])

    @staticmethod
    def obj_func_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        S1 = 40 - 40/500 * (x[0] + x[5] + x[8] + x[11])
        S2 = 35 - 35/400 * (x[3] + x[6] + x[9] + x[12])
        S3 = 32 - 32/600 * (x[4] + x[7] + x[10] + x[13])
        return (15 - S1)(x[0] + x[5]) + (15 - S2)(x[2] + x[6]) + (15 - S3)(x[3] + x[6])

    @staticmethod
    def obj_func_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        S1 = 40 - 40/500 * (x[0] + x[5] + x[8] + x[11])
        S2 = 35 - 35/400 * (x[3] + x[6] + x[9] + x[12])
        S3 = 32 - 32/600 * (x[4] + x[7] + x[10] + x[13])
        return (15 - S1)(x[6] + x[10]) + (15 - S2)(x[9] + x[12]) + (15 - S3)(x[10] + x[12])
   
    @staticmethod
    def obj_func_der_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([15 - 40 - 80/500*x[0],
                         15 - 40 - 80/500*x[1],
                         15 - 35 - 70/400*x[2],
                         15 - 35 - 70/400*x[3],
                         15 - 32 - 64/600*x[4],
                         15 - 32 - 64/600*x[5]])
   
   
    @staticmethod
    def obj_func_der_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([15 - 40 - 80/500*x[6],
                         15 - 40 - 80/500*x[7],
                         15 - 35 - 70/400*x[8],
                         15 - 35 - 70/400*x[9],
                         15 - 32 - 64/600*x[10],
                         15 - 32 - 64/600*x[11]])

    @staticmethod
    def g0(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return x[0] + x[1] + x[2] - 100
   
    @staticmethod
    def g1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return x[3] + x[4] + x[5] - 50
   
    @staticmethod
    def g2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return x[6] + x[7] + x[8] - 100
   
    @staticmethod
    def g3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return x[9] + x[10] + x[11] - 50
   
    @staticmethod
    def g4(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        S1 = 40 - 40/500 * (x[0] + x[5] + x[8] + x[11])
        S2 = 35 - 35/400 * (x[3] + x[6] + x[9] + x[12])
        S3 = 32 - 32/600 * (x[4] + x[7] + x[10] + x[13])
        return abs(S1 - S2) - 1
   
    @staticmethod
    def g5(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        S1 = 40 - 40/500 * (x[0] + x[5] + x[8] + x[11])
        S2 = 35 - 35/400 * (x[3] + x[6] + x[9] + x[12])
        S3 = 32 - 32/600 * (x[4] + x[7] + x[10] + x[13])
        return abs(S1 - S3) - 1

    @staticmethod
    def g6(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        S1 = 40 - 40/500 * (x[0] + x[5] + x[8] + x[11])
        S2 = 35 - 35/400 * (x[3] + x[6] + x[9] + x[12])
        S3 = 32 - 32/600 * (x[4] + x[7] + x[10] + x[13])
        return abs(S2 - S3) - 1
   
    @staticmethod
    def g0_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array ([1, 1, 1, 0, 0, 0]).reshape(-1, 1)
   
    @staticmethod
    def g1_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array ([0, 1, 1, 1, 0, 0]).reshape(-1, 1)
   
    @staticmethod
    def g2_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array ([0, 0, 1, 1, 1, 0]).reshape(-1, 1)
   
    @staticmethod
    def g3_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array ([0, 0, 0, 1, 1, 1]).reshape(-1, 1)
   
    @staticmethod
    def g4_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array ([-40/500, 35/400, 0, -40/500, -35/400, 0,
                          40/500, -35,400, 0, -40/500, -35/400, 0]).reshape(-1, 1)

    @staticmethod
    def g5_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array ([-40/500, 0, -32/600, -40/500, 0, -32/600, 
                          -40/500, 0, -32/600, -40/500, 0, -32,600]).reshape(-1, 1)
   
    @staticmethod
    @staticmethod
    def g6_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array ([0, -35/400, -32/600, 0, -35/400, -32/600,
                          0, -35,400, -32/600, 0, -35/400, -32/600]).reshape(-1, 1)