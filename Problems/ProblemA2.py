import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt

class A2:
    @staticmethod
    def paper_solution():
        value_1 = [0.29962894677774, 0.00997828224734, 0.00997828224734,
                    0.00997828224734, 0.59852469355630, 0.02187270661760, 
                    0.00999093169361, 0.00999093169361, 0.00999093169361, 
                    0.00999093169361]

        value_2 = [0.29962898846513, 0.00997828313762, 0.00997828313762, 
                   0.00997828313762,  0.59745624992082, 0.02220301920403, 
                   0.01013441012117, 0.01013441012117, 0.01013441012117, 
                   0.01013441012117]

        return [value_1, value_2]
    
    @staticmethod
    def define_players():
        player_vector_sizes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        player_objective_functions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        player_constraints = [[1, 2], (0, 3), (0, 4), (0, 5), (0, 6, 12), (0, 7, 12), (0, 8), (0, 9), (0, 10), (0, 11)]
        bounds = [(0.3, 0.5), (0.01, 1), (0.01, 1), (0.01, 1), (0.01, 1), (0.01, 1), (0.01, 1), (0.01, 1), (0.01, 0.06), (0.01, 0.05)]
        bounds_training = [(0.3, 0.5), (0.01, 1), (0.01, 1), (0.01, 1), (0.01, 1), (0.01, 1), (0.01, 1), (0.01, 1), (0.01, 0.06), (0.01, 0.05)]
        return [player_vector_sizes, player_objective_functions, player_constraints, bounds, bounds_training]
    
    @staticmethod
    def objective_functions():
        return [A2.obj_func_1, A2.obj_func_2, A2.obj_func_3, A2.obj_func_4, A2.obj_func_5, A2.obj_func_6, A2.obj_func_7, A2.obj_func_8, A2.obj_func_9, A2.obj_func_10]

    @staticmethod
    def objective_function_derivatives():
        return [A2.obj_func_der_1, A2.obj_func_der_2, A2.obj_func_der_3, A2.obj_func_der_4, A2.obj_func_der_5, A2.obj_func_der_6, A2.obj_func_der_7, A2.obj_func_der_8, A2.obj_func_der_9, A2.obj_func_der_10]

    @staticmethod
    def constraints():
        return [A2.g0, A2.g1, A2.g2, A2.g3, A2.g4, A2.g5, A2.g6, A2.g7, A2.g8, A2.g9]

    @staticmethod
    def constraint_derivatives():
        return [A2.g0_der, A2.g1_der, A2.g2_der, A2.g3_der, A2.g4_der, A2.g5_der, A2.g6_der, A2.g7_der, A2.g8_der, A2.g9_der]

    B = 1
    def sum_x(x):
        return sum(x)
    
    @staticmethod
    def obj_func_1(x):  
        S = A2.sum_x(x)
        return -x[0] * S * (1 - S / B)
    
    @staticmethod
    def obj_func_2(x):  
        S = sum_x(x)
        return -x[1] * S * (1 - S / B)**2

    @staticmethod
    def obj_func_3(x): 
        S = sum_x(x)
        return -x[2] * S * (1 - S / B)**2
    @staticmethod
    def obj_func_4(x): 
        S = sum_x(x)
        return -x[3] * S * (1 - S / B)**2
    
    @staticmethod
    def obj_func_5(x): 
        S = sum_x(x)
        return -x[4] * S * (1 - S / B)**2
    
    @staticmethod
    def obj_func_6(x):  
        S = sum_x(x)
        return -x[5] * S * (1 - S / B)

    @staticmethod
    def obj_func_7(x):  
       S = sum_x(x)
       return -x[6] * S * (1 - S / B)

    @staticmethod
    def obj_func_8(x):  
       S = sum_x(x)
       return -x[7] * S * (1 - S / B)

    @staticmethod
    def obj_func_9(x):  
       S = sum_x(x)
       return -x[8] * S * (1 - S / B)

    @staticmethod
    def obj_func_10(x):  
       S = sum_x(x)
       return -x[9] * S * (1 - S / B)

    @staticmethod
    def layer_objective_functions():
     return [
        A2.obj_func_1,
        A2.obj_func_2,
        A2.obj_func_3,
        A2.obj_func_4,
        A2.obj_func_5,
        A2.obj_func_6,
        A2.obj_func_7,
        A2.obj_func_8,
        A2.obj_func_9,
        A2.obj_func_10
    ]

    @staticmethod
    def obj_func_der_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        S = np.sum(x)
        d = -S * (1 - S) - x[0] * (1 - 2 * S)
        return np.array([[d]])

    @staticmethod
    def obj_func_der_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        S = np.sum(x)
        d = -S * (1 - S)**2 - x[1] * 2 * (1 - S) * (1 - 2 * S)
        return np.array([[d]])

    @staticmethod
    def obj_func_der_3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        S = np.sum(x)
        d = -S * (1 - S)**2 - x[2] * 2 * (1 - S) * (1 - 2 * S)
        return np.array([[d]])

    @staticmethod
    def obj_func_der_4(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        S = np.sum(x)
        d = -S * (1 - S)**2 - x[3] * 2 * (1 - S) * (1 - 2 * S)
        return np.array([[d]])

    @staticmethod
    def obj_func_der_5(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        S = np.sum(x)
        d = -S * (1 - S)**2 - x[4] * 2 * (1 - S) * (1 - 2 * S)
        return np.array([[d]])

    @staticmethod
    def obj_func_der_6(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        S = np.sum(x)
        d = -S * (1 - S) - x[5] * (1 - 2 * S)
        return np.array([[d]])

    @staticmethod
    def obj_func_der_7(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        S = np.sum(x)
        d = -S * (1 - S) - x[6] * (1 - 2 * S)
        return np.array([[d]])

    @staticmethod
    def obj_func_der_8(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        S = np.sum(x)
        d = -S * (1 - S) - x[7] * (1 - 2 * S)
        return np.array([[d]])

    @staticmethod
    def obj_func_der_9(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        S = np.sum(x)
        d = -S * (1 - S) - x[8] * (1 - 2 * S)
        return np.array([[d]])

    @staticmethod
    def obj_func_der_10(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        S = np.sum(x)
        d = -S * (1 - S) - x[9] * (1 - 2 * S)
        return np.array([[d]])


    @staticmethod
    def g0(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Player 1: bounds 0.3 ≤ x0 ≤ 0.5
        return np.array([
            0.3 - x[0],   # x0 ≥ 0.3 -> 0.3 - x[0] ≤ 0
            x[0] - 0.5    # x0 ≤ 0.5 -> x[0] - 0.5 ≤ 0
        ])

    @staticmethod
    def g1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Player 2: x1 ≥ 0.01, ∑x ≤ 1
        return np.array([
            0.01 - x[1],         # x1 ≥ 0.01
            np.sum(x) - 1        # ∑x ≤ 1
        ])

    @staticmethod
    def g2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Player 3: x2 ≥ 0.01, ∑x ≤ 1
        return np.array([
            0.01 - x[2],
            np.sum(x) - 1
        ])

    @staticmethod
    def g3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Player 4: x3 ≥ 0.01, ∑x ≤ 1
        return np.array([
            0.01 - x[3],
            np.sum(x) - 1
        ])

    @staticmethod
    def g4(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Player 5: x4 ≥ 0.01, 0.99 ≤ ∑x ≤ 1
        return np.array([
            0.01 - x[4],        # x4 ≥ 0.01
            0.99 - np.sum(x),   # ∑x ≥ 0.99 -> 0.99 - ∑x ≤ 0
            np.sum(x) - 1       # ∑x ≤ 1
        ])

    @staticmethod
    def g5(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Player 6: x5 ≥ 0.01, 0.99 ≤ ∑x ≤ 1
        return np.array([
            0.01 - x[5],
            0.99 - np.sum(x),
            np.sum(x) - 1
        ])

    @staticmethod
    def g6(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Player 7: x6 ≥ 0.01, ∑x ≤ 1
        return np.array([
            0.01 - x[6],
            np.sum(x) - 1
        ])

    @staticmethod
    def g7(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Player 8: x7 ≥ 0.01, ∑x ≤ 1
        return np.array([
            0.01 - x[7],
            np.sum(x) - 1
        ])

    @staticmethod
    def g8(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Player 9: x8 ≥ 0.01, x8 ≤ 0.06, ∑x ≤ 1
        return np.array([
            0.01 - x[8],
            x[8] - 0.06,
            np.sum(x) - 1
        ])

    @staticmethod
    def g9(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Player 10: x9 ≥ 0.01, x9 ≤ 0.05, ∑x ≤ 1
        return np.array([
            0.01 - x[9],
            x[9] - 0.05,
            np.sum(x) - 1
        ])
    @staticmethod
    def g0_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # x0 ∈ [0.3, 0.5] → constraints: -x0 + 0.3 ≤ 0, x0 - 0.5 ≤ 0
        G = np.zeros((2, 10))
        G[0, 0] = -1
        G[1, 0] = 1
        return G.T

    @staticmethod
    def g1_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        G = np.zeros((2, 10))
        G[0, 1] = -1          # x1 ≥ 0.01 ⇒ -x1 + 0.01 ≤ 0
        G[1, :] = 1           # sum(x) ≤ 1 ⇒ ∂/∂x = [1,...,1]
        return G.T

    @staticmethod
    def g2_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        G = np.zeros((2, 10))
        G[0, 2] = -1
        G[1, :] = 1
        return G.T

    @staticmethod
    def g3_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        G = np.zeros((2, 10))
        G[0, 3] = -1
        G[1, :] = 1
        return G.T

    @staticmethod
    def g4_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        G = np.zeros((3, 10))
        G[0, 4] = -1                      # x4 ≥ 0.01
        G[1, :] = -1                     # sum(x) ≥ 0.99 ⇒ -sum(x) + 0.99 ≤ 0
        G[2, :] = 1                      # sum(x) ≤ 1
        return G.T

    @staticmethod
    def g5_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        G = np.zeros((3, 10))
        G[0, 5] = -1
        G[1, :] = -1
        G[2, :] = 1
        return G.T

    @staticmethod
    def g6_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        G = np.zeros((2, 10))
        G[0, 6] = -1
        G[1, :] = 1
        return G.T

    @staticmethod
    def g7_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        G = np.zeros((2, 10))
        G[0, 7] = -1
        G[1, :] = 1
        return G.T

    @staticmethod
    def g8_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        G = np.zeros((3, 10))
        G[0, 8] = -1                  # x8 ≥ 0.01
        G[1, 8] = 1                   # x8 ≤ 0.06
        G[2, :] = 1                   # sum(x) ≤ 1
        return G.T

    @staticmethod
    def g9_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        G = np.zeros((3, 10))
        G[0, 9] = -1                  # x9 ≥ 0.01
        G[1, 9] = 1                   # x9 ≤ 0.05
        G[2, :] = 1                   # sum(x) ≤ 1
        return G.T