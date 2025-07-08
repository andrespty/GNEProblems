import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt

class A16:
    #Problem parameters
    N = 5
    c = np.array([10, 8, 6, 4, 2], dtype=float)
    K = np.array([5] * N, dtype=float)
    B = np.array([1.2, 1.1, 1.0, 0.9, 0.8], dtype=float)
    Q = 100  #Max total output

    @staticmethod
    def paper_solution() -> List[npt.NDArray[np.float64]]:
        value_1 = np.array([10.403965, 13.035817, 15.407354, 17.381556, 18.771308]) #Q = 75
        value_2 = np.array([14.050088, 17.798379, 20.907187, 23.111429, 24.132916]) #Q = 100
        value_3 = np.array([23.588799, 28.684248, 32.021533, 33.287258, 32.418182]) #Q = 150
        value_4 = np.array([35.785329, 40.748959, 42.802485, 41.966381, 38.696846]) #Q = 200
        value_5 = np.array([36.912, 41.842, 43.705, 42.665, 39.182]) #Q = 204.306
        return [value_1, value_2, value_3, value_4, value_5]

    @staticmethod
    def define_players():
      player_vector_size = [1, 1, 1, 1, 1]
      player_objective_functions = [0, 1, 2, 3, 4] #Seperate objective function for each player
      player_constraints = [0, 0, 0, 0, 0] #Seperate constraint function for each player
      bounds = [(0, 100)] * 5
      bounds_training = [(0, 100)] * 5
      return [player_vector_size, player_objective_functions, player_constraints, bounds, bounds_training]

    @staticmethod
    def objective_functions() -> List[Callable[[float, float, float], float]]:
        return [A16.player_objective_i(i) for i in range(A16.N)]

    @staticmethod
    def player_objective_i(i: int) -> Callable[[float, float, float], float]:
        def obj(q_i: float, total_q: float, d: float) -> float:
            #price and cost calculations
            p = 5000**(1 / 1.1) * total_q**(-1 / 1.1)
            B_i = A16.B[i]
            K_i = A16.K[i]
            c_i = A16.c[i]
            cost = c_i * q_i + (B_i / (B_i + 1)) * K_i**(-1 / B_i) * q_i**((B_i + 1) / B_i)
            return -(q_i * p - cost - d * q_i)
        return obj

    @staticmethod
    def objective_function_derivatives() -> List[Callable[[float, float, float], float]]:
        return [A16.player_objective_derivative_i(i) for i in range(A16.N)]

    @staticmethod
    def player_objective_derivative_i(i: int) -> Callable[[float, float, float], float]:
        def grad(q_i: float, total_q: float, d: float) -> float:
            # Inline price derivative and cost derivative
            price = 5000**(1 / 1.1) * total_q**(-1 / 1.1)
            price_deriv = -(1 / 1.1) * 5000**(1 / 1.1) * total_q**(-1 / 1.1 - 1)
            dp = price + q_i * price_deriv

            B_i = A16.B[i]
            K_i = A16.K[i]
            c_i = A16.c[i]
            dc = c_i + K_i**(-1 / B_i) * q_i**(1 / B_i)
            return -(dp - dc - d)
        return grad
   
    #constraint stuff

    @staticmethod
    def constraints() -> List[Callable[[npt.NDArray[np.float64]], float]]:
        return [A16.g0]

    @staticmethod
    def constraint_derivatives() -> List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]]:
        return [A16.g0_der]

    @staticmethod
    def g0(q: npt.NDArray[np.float64]) -> float:
        return np.sum(q) - A16.Q

    @staticmethod
    def g0_der(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.ones_like(q)