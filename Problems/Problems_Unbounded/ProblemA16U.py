import numpy as np
import numpy.typing as npt
from typing import Callable, List
 
class A16U:
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
        n = 5
        player_vector_size = [1 for _ in range(n)]
        player_objective_functions = [0, 1, 2, 3, 4]
        player_constraints = [[0, 1] for _ in range(n)]
        return [player_vector_size, player_objective_functions, player_constraints]

    @staticmethod
    def objective_functions():
        return [A16U.obj_func_0, A16U.obj_func_1, A16U.obj_func_2, A16U.obj_func_3, A16U.obj_func_4]

    @staticmethod
    def objective_function_derivatives():
        return [A16U.obj_func_der_0, A16U.obj_func_der_1, A16U.obj_func_der_2, A16U.obj_func_der_3, A16U.obj_func_der_4]

    @staticmethod
    def constraints():
        return [A16U.g0, A16U.g100]

    @staticmethod
    def constraint_derivatives() -> List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]]:
        return [A16U.g0_der, A16U.gx_der]

    # --- Objective Functions ---

    @staticmethod
    def obj_func(
            x_i:npt.NDArray[np.float64],
            c_i:npt.NDArray[np.float64],
            K_i:npt.NDArray[np.float64],
            delta_i:npt.NDArray[np.float64],
            S: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        gamma = 1.1
        f_i = c_i * x_i + (delta_i/(1 + delta_i)) * np.pow(K_i, -(1/delta_i)) * np.pow(x_i, (1+delta_i)/delta_i)
        obj = f_i - np.pow(5000, 1/gamma) * x_i * np.pow(S, -1/gamma)
        return obj
 
    @staticmethod
    def obj_func_0(x: npt.NDArray[np.float64]) -> float:
        c1 = 10
        K1 = 5
        delta1 = 1.2
        S = np.sum(np.concatenate(x))
        return A16U.obj_func(x[0], c1, K1, delta1, S)

    @staticmethod
    def obj_func_1(x: npt.NDArray[np.float64]) -> float:
        c2 = 8
        K2 = 5
        delta2 = 1.1
        S = np.sum(np.concatenate(x))
        return A16U.obj_func(x[1], c2, K2, delta2, S)

    @staticmethod
    def obj_func_2(x: npt.NDArray[np.float64]) -> float:
        c3 = 6
        K3 = 5
        delta3 = 1.0
        S = np.sum(np.concatenate(x))
        return A16U.obj_func(x[2], c3, K3, delta3, S)

    @staticmethod
    def obj_func_3(x: npt.NDArray[np.float64]) -> float:
        c4 = 4
        K4 = 5
        delta4 = 0.9
        S = np.sum(np.concatenate(x))
        return A16U.obj_func(x[3], c4, K4, delta4, S)

    @staticmethod
    def obj_func_4(x: npt.NDArray[np.float64]) -> float:
        c5 = 2
        K5 = 5
        delta5 = 0.8
        S = np.sum(np.concatenate(x))
        return A16U.obj_func(x[4], c5, K5, delta5, S)

    # --- Objective Derivatives ---
    @staticmethod
    def obj_func_der(
            x_i:npt.NDArray[np.float64],
            c_i:npt.NDArray[np.float64],
            K_i:npt.NDArray[np.float64],
            delta_i:npt.NDArray[np.float64],
            S: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        gamma = 1.1
        f_i = c_i + np.pow(K_i, -1/delta_i) * np.pow(x_i, (1/delta_i))
        obj_der = f_i - np.pow(5000, 1/gamma) * ( np.pow(S, -1/gamma) - (x_i/gamma) * np.pow(S, (-1/gamma)-1 ) )
        return obj_der

    @staticmethod
    def obj_func_der_0(x):
        c1 = 10
        K1 = 5
        delta1 = 1.2
        S = np.sum(np.concatenate(x))
        return A16U.obj_func(x[0], c1, K1, delta1, S)
 
    @staticmethod
    def obj_func_der_1(x):
        c2 = 8
        K2 = 5
        delta2 = 1.1
        S = np.sum(np.concatenate(x))
        return A16U.obj_func(x[1], c2, K2, delta2, S)
 
    @staticmethod
    def obj_func_der_2(x):
        c3 = 6
        K3 = 5
        delta3 = 1.0
        S = np.sum(np.concatenate(x))
        return A16U.obj_func(x[2], c3, K3, delta3, S)
 
    @staticmethod
    def obj_func_der_3(x):
        c4 = 4
        K4 = 5
        delta4 = 0.9
        S = np.sum(np.concatenate(x))
        return A16U.obj_func(x[3], c4, K4, delta4, S)
 
    @staticmethod
    def obj_func_der_4(x):
        c5 = 2
        K5 = 5
        delta5 = 0.8
        S = np.sum(np.concatenate(x))
        return A16U.obj_func(x[4], c5, K5, delta5, S)
 

    # --- Constraints ---
    @staticmethod
    def g0(x: list[npt.NDArray[np.float64]]) -> float:
        return 0 - np.concatenate(x).reshape(-1, 1)

    @staticmethod
    def g75(x: list[npt.NDArray[np.float64]]) -> float:
        P= 75
        x = np.concatenate(x).reshape(-1, 1)
        return np.sum(x) - P

    @staticmethod
    def g100(x: list[npt.NDArray[np.float64]]) -> float:
        P = 100
        x = np.concatenate(x).reshape(-1, 1)
        return np.sum(x) - P

    @staticmethod
    def g150(x: list[npt.NDArray[np.float64]]) -> float:
        P = 150
        x = np.concatenate(x).reshape(-1, 1)
        return np.sum(x) - P

    @staticmethod
    def g200(x: list[npt.NDArray[np.float64]]) -> float:
        P = 200
        x = np.concatenate(x).reshape(-1, 1)
        return np.sum(x) - P
 
    @staticmethod
    def g0_der(x: npt.NDArray[np.float64]) -> float:
        return -1

    @staticmethod
    def gx_der(x: list[npt.NDArray[np.float64]]) -> float:
        return 1
