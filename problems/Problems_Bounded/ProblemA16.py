import numpy as np
import numpy.typing as npt
from typing import Callable, List
 
class A16:
   # Problem parameters
    N = 5
    c = np.array([10, 8, 6, 4, 2], dtype=float)
    K = np.array([5] * N, dtype=float)
    B = np.array([1.2, 1.1, 1.0, 0.9, 0.8], dtype=float)
    Q = 200  # Max total output

    @staticmethod
    def paper_solution() -> List[npt.NDArray[np.float64]]:
        value_1 = np.array([10.403965, 13.035817, 15.407354, 17.381556, 18.771308])  # 75
        value_2 = np.array([14.050088, 17.798379, 20.907187, 23.111429, 24.132916])  # 100
        value_3 = np.array([23.588799, 28.684248, 32.021533, 33.287258, 32.418182])  # 150
        value_4 = np.array([35.785329, 40.748959, 42.802485, 41.966381, 38.696846])  # 200
        value_5 = np.array([36.912, 41.842, 43.705, 42.665, 39.182])  # 204.306
        return [value_1, value_2, value_3, value_4, value_5]
 
    @staticmethod
    def define_players():
        player_vector_size = [1, 1, 1, 1, 1]
        player_objective_functions = [0, 1, 2, 3, 4]
        player_constraints = [[0], [0], [0], [0], [0]]
        bounds = [(0, A16.Q)] * 5 + [(0, 100)]
        bounds_training = [(0, A16.Q)] * 5 + [(0, 1000)]
        return [player_vector_size, player_objective_functions, player_constraints, bounds, bounds_training]
 
    # --- Objective Functions ---
 
    @staticmethod
    def obj_func_0(x: npt.NDArray[np.float64]) -> float:
        q_i, d = x[0], x[-1]
        total_q = np.sum(x[:-1])
        p = 5000**(1 / 1.1) * total_q**(-1 / 1.1)
        cost = 10 * q_i + (1.2 / (1.2 + 1)) * 5**(-1 / 1.2) * q_i**((1.2 + 1) / 1.2)
        return -(q_i * p - cost - d * q_i)
 
    @staticmethod
    def obj_func_1(x):
        q_i, d = x[1], x[-1]
        total_q = np.sum(x[:-1])
        p = 5000**(1 / 1.1) * total_q**(-1 / 1.1)
        cost = 8 * q_i + (1.1 / (1.1 + 1)) * 5**(-1 / 1.1) * q_i**((1.1 + 1) / 1.1)
        return -(q_i * p - cost - d * q_i)
 
    @staticmethod
    def obj_func_2(x):
        q_i, d = x[2], x[-1]
        total_q = np.sum(x[:-1])
        p = 5000**(1 / 1.1) * total_q**(-1 / 1.1)
        cost = 6 * q_i + (1.0 / (1.0 + 1)) * 5**(-1 / 1.0) * q_i**((1.0 + 1) / 1.0)
        return -(q_i * p - cost - d * q_i)
 
    @staticmethod
    def obj_func_3(x):
        q_i, d = x[3], x[-1]
        total_q = np.sum(x[:-1])
        p = 5000**(1 / 1.1) * total_q**(-1 / 1.1)
        cost = 4 * q_i + (0.9 / (0.9 + 1)) * 5**(-1 / 0.9) * q_i**((0.9 + 1) / 0.9)
        return -(q_i * p - cost - d * q_i)
 
    @staticmethod
    def obj_func_4(x):
        q_i, d = x[4], x[-1]
        total_q = np.sum(x[:-1])
        p = 5000**(1 / 1.1) * total_q**(-1 / 1.1)
        cost = 2 * q_i + (0.8 / (0.8 + 1)) * 5**(-1 / 0.8) * q_i**((0.8 + 1) / 0.8)
        return -(q_i * p - cost - d * q_i)
 
    @staticmethod
    def objective_functions():
        return [A16.obj_func_0, A16.obj_func_1, A16.obj_func_2, A16.obj_func_3, A16.obj_func_4]
 
    # --- Objective Derivatives ---
 
    @staticmethod
    def obj_func_der_0(x):
        q_i, d = x[0], x[-1]
        total_q = np.sum(x[:-1])
        price = 5000**(1 / 1.1) * total_q**(-1 / 1.1)
        price_deriv = -(1 / 1.1) * 5000**(1 / 1.1) * total_q**(-1 / 1.1 - 1)
        dp = price + q_i * price_deriv
        dc = 10 + 5**(-1 / 1.2) * q_i**(1 / 1.2)
        return -(dp - dc - d)
 
    @staticmethod
    def obj_func_der_1(x):
        q_i, d = x[1], x[-1]
        total_q = np.sum(x[:-1])
        price = 5000**(1 / 1.1) * total_q**(-1 / 1.1)
        price_deriv = -(1 / 1.1) * 5000**(1 / 1.1) * total_q**(-1 / 1.1 - 1)
        dp = price + q_i * price_deriv
        dc = 8 + 5**(-1 / 1.1) * q_i**(1 / 1.1)
        return -(dp - dc - d)
 
    @staticmethod
    def obj_func_der_2(x):
        q_i, d = x[2], x[-1]
        total_q = np.sum(x[:-1])
        price = 5000**(1 / 1.1) * total_q**(-1 / 1.1)
        price_deriv = -(1 / 1.1) * 5000**(1 / 1.1) * total_q**(-1 / 1.1 - 1)
        dp = price + q_i * price_deriv
        dc = 6 + 5**(-1 / 1.0) * q_i**(1 / 1.0)
        return -(dp - dc - d)
 
    @staticmethod
    def obj_func_der_3(x):
        q_i, d = x[3], x[-1]
        total_q = np.sum(x[:-1])
        price = 5000**(1 / 1.1) * total_q**(-1 / 1.1)
        price_deriv = -(1 / 1.1) * 5000**(1 / 1.1) * total_q**(-1 / 1.1 - 1)
        dp = price + q_i * price_deriv
        dc = 4 + 5**(-1 / 0.9) * q_i**(1 / 0.9)
        return -(dp - dc - d)
 
    @staticmethod
    def obj_func_der_4(x):
        q_i, d = x[4], x[-1]
        total_q = np.sum(x[:-1])
        price = 5000**(1 / 1.1) * total_q**(-1 / 1.1)
        price_deriv = -(1 / 1.1) * 5000**(1 / 1.1) * total_q**(-1 / 1.1 - 1)
        dp = price + q_i * price_deriv
        dc = 2 + 5**(-1 / 0.8) * q_i**(1 / 0.8)
        return -(dp - dc - d)
 
    @staticmethod
    def objective_function_derivatives():
        return [A16.obj_func_der_0, A16.obj_func_der_1, A16.obj_func_der_2, A16.obj_func_der_3, A16.obj_func_der_4]
 
    # --- Constraints ---
 
    @staticmethod
    def constraints() -> List[Callable[[npt.NDArray[np.float64]], float]]:
        return [A16.g0]
 
    @staticmethod
    def constraint_derivatives() -> List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]]:
        return [A16.g0_der]
 
    @staticmethod
    def g0(x: npt.NDArray[np.float64]) -> float:
        return np.sum(x) - A16.Q
 
    @staticmethod
    def g0_der(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        grad = np.zeros_like(q)
        grad[:-1] = 1.0  # d/dq_i of sum(q[:-1]) is 1 for all i < N
        return 1
