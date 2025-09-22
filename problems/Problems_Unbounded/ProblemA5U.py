import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt

class A5U:
    @staticmethod
    def paper_solution():
        value_1 = [-0.00006229891126, 0.20279012064850, -0.00003469558295,
                       -0.00028322020054, 0.07258934064261,
                       0.02531280162415, -0.00007396699835]

        value_2 = [-0.00006229891126, 0.20279012064850, -0.00003469558295,
                       -0.00028322020054, 0.07258934064261,
                       0.02531280162415, -0.00007396699835]

        value_3 = [-0.00006229910314, 0.20279011130836, -0.00003469562269,
                       -0.00028322027018, 0.07258933626181,
                       0.02531280221816, -0.00007396699882]
        return [value_1, value_2, value_3]

    @staticmethod
    def define_players():
        player_vector_sizes = [3, 2, 2]
        player_objective_functions = [0, 1, 2]
        player_constraints = [[0, 1, 4, 5],[2, 4, 5],[3, 4, 5]]
        return [player_vector_sizes, player_objective_functions, player_constraints]

    @staticmethod
    def objective_functions():
        return [A5U.obj_func_1, A5U.obj_func_2, A5U.obj_func_3]

    @staticmethod
    def objective_function_derivatives():
        return [A5U.obj_func_der_1, A5U.obj_func_der_2, A5U.obj_func_der_3]

    @staticmethod
    def constraints():
        return [A5U.g0, A5U.g1,A5U.g2, A5U.g3, A5U.g4, A5U.g5]

    @staticmethod
    def constraint_derivatives():
        return [A5U.g0_der, A5U.g1_der,A5U.g2_der, A5U.g3_der, A5U.g4_der, A5U.g5_der]

    # Define Functions below
    A1 = np.array([
        [20, 6, 0],
        [6, 6, -1],
        [0, -1, 8]
    ])
    B1 = np.array([[-1, -2, -4, -3], [0, -3, 0, -4], [0, 1, 9, 6]])
    b1 = np.array([[1], [-1], [1]])
    A2 = np.array([
        [11, 1],
        [1, 7]
    ])
    B2 = np.array([[-1, 0, 0, -7, 4], [-2, -3, 1, 4, 11]])
    b2 = np.array([[1], [0]])
    A3 = np.array([
        [28, 14],
        [14, 29]
    ])
    B3 = np.array([[-4, 0, 9, -7, 4], [-3, -4, 6, 4, 11]])
    b3 = np.array([[-1], [2]])


    @staticmethod
    def obj_func(
            x: npt.NDArray[np.float64],
            x_ni: npt.NDArray[np.float64],
            A: npt.NDArray[np.float64],
            B: npt.NDArray[np.float64],
            b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # x: actions vector (d_i, 1)
        # A: constant matrix (m, n)
        # B: constant matrix (m, sum(d_{-i}))
        # b: constant vector (m, 1)
        obj = 0.5 * x.T @ A @ x + x.T @ (B @ x_ni + b)
        return obj

    @staticmethod
    def obj_func_1(x: list[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
        """
        Parameters
        ----------
        x : list of numpy.ndarray
            List of NumPy arrays (dtype float64).
            - x[0], x[1], and x[2] are required.
            - Each element should be compatible with reshaping into column vectors.

        Returns
        -------
        float
        """
        x1 = x[0]
        x_n1 = np.vstack((x[1], x[2]))
        return A5U.obj_func(x1, x_n1, A5U.A1, A5U.B1, A5U.b1)

    @staticmethod
    def obj_func_2(x: list[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
        """
        Parameters
        ----------
        x : list of numpy.ndarray
            List of NumPy arrays (dtype float64).
            - x[0], x[1], and x[2] are required.
            - Each element should be compatible with reshaping into column vectors.

        Returns
        -------
        float
        """
        x2 = x[1]
        x_n2 = np.vstack((x[0], x[2]))
        return A5U.obj_func(x2, x_n2, A5U.A2, A5U.B2, A5U.b2)

    @staticmethod
    def obj_func_3(x: list[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
        """
        Parameters
        ----------
        x : list of numpy.ndarray
            List of NumPy arrays (dtype float64).
            - x[0], x[1], and x[2] are required.
            - Each element should be compatible with reshaping into column vectors.

        Returns
        -------
        float
        """
        x3 = x[2]
        x_n3 = np.vstack((x[0], x[1]))
        return A5U.obj_func(x3, x_n3, A5U.A3, A5U.B3, A5U.b3)


    @staticmethod
    def obj_func_der(
            x: npt.NDArray[np.float64],
            x_ni: npt.NDArray[np.float64],
            A: npt.NDArray[np.float64],
            B: npt.NDArray[np.float64],
            b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # x: actions vector (d_i, 1)
        # A: constant matrix (m, n)
        # B: constant matrix (m, sum(d_{-i}))
        # b: constant vector (m, 1)
        obj = A @ x + B @ x_ni + b
        return obj

    @staticmethod
    def obj_func_der_1(x: list[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x_n1 = np.vstack((x[1], x[2]))
        return A5U.obj_func_der(x1, x_n1, A5U.A1, A5U.B1, A5U.b1)

    @staticmethod
    def obj_func_der_2(x: list[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
        x2 = x[1]
        x_n2 = np.vstack((x[0], x[2]))
        return A5U.obj_func_der(x2, x_n2, A5U.A2, A5U.B2, A5U.b2)

    @staticmethod
    def obj_func_der_3(x: list[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
        x3 = x[2]
        x_n3 = np.vstack((x[0], x[1]))
        return A5U.obj_func_der(x3, x_n3, A5U.A3, A5U.B3, A5U.b3)

    @staticmethod
    def g0(x):
        x1, x2, x3 = x
        return np.sum(x1) - 20

    @staticmethod
    def g1(x):
        x1, x2, x3 = x
        return (x1[0] + x1[1] - x1[2] - x2[0] + x3[1] - 5)[0]

    @staticmethod
    def g2(x):
        x1, x2, x3 = x
        return (x2[0] - x2[1] - x1[1] - x1[2] + x3[0] - 7)[0]

    @staticmethod
    def g3(x):
        x1, x2, x3 = x
        return (x3[1] - x1[0] - x1[2] + x2[0] - 4)[0]
    
    @staticmethod
    def g4(x):
        x1, x2, x3 = x
        return 0 - np.vstack((x1.reshape(-1, 1), x2.reshape(-1, 1), x3.reshape(-1, 1)))

    @staticmethod
    def g5(x):
        x1, x2, x3 = x
        return  np.vstack((x1.reshape(-1, 1), x2.reshape(-1, 1), x3.reshape(-1, 1))) -10 

    # partial g0 / partial x1
    @staticmethod
    def g0_der(x: list[np.ndarray]) -> np.ndarray:
        grad = np.zeros((7, 1))
        grad[0] = 1
        grad[1] = 1
        grad[2] = 1
        grad[3] = 0
        grad[4] = 0
        grad[5] = 0
        grad[6] = 0
        return grad

    # partial g1 / partial x1
    @staticmethod
    def g1_der(x: list[np.ndarray]) -> np.ndarray:
        grad = np.zeros((7, 1))
        grad[0] = 1
        grad[1] = 1
        grad[2] = -1
        grad[3] = -1
        grad[4] = 0
        grad[5] = 0
        grad[6] = 1
        return grad

    # partial g2 / partial x2
    @staticmethod
    def g2_der(x: list[np.ndarray]) -> np.ndarray:
        grad = np.zeros((7, 1))
        grad[0] = 0
        grad[1] = -1
        grad[2] = -1
        grad[3] = 1
        grad[4] = -1
        grad[5] = 1
        grad[6] = 0
        return grad

    # partial g3 / partial x3
    @staticmethod
    def g3_der(x: list[np.ndarray]) -> np.ndarray:
        grad = np.zeros((7, 1))
        grad[0] = -1
        grad[1] = 0
        grad[2] = -1
        grad[3] = 1
        grad[4] = 0
        grad[5] = 0
        grad[6] = 1
        return grad
    
    @staticmethod
    def g4_der(x: list[np.ndarray]) -> np.ndarray:
        grad = np.zeros((7, 1))
        grad[0] = -1
        grad[1] = -1
        grad[2] = -1
        grad[3] = -1
        grad[4] = -1
        grad[5] = -1
        grad[6] = -1
        return grad
    
    @staticmethod
    def g5_der(x: list[np.ndarray]) -> np.ndarray:
        grad = np.zeros((7, 1))
        grad[0] = 1
        grad[1] = 1
        grad[2] = 1
        grad[3] = 1
        grad[4] = 1
        grad[5] = 1
        grad[6] = 1
        return grad
    