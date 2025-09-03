import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt

from gne_solver.misc import *


class A3dev:

    @staticmethod
    def paper_solution():
        value_1 = [-0.38046562696258, -0.12266997083581, -0.99322817120517,
                   0.39034789080544, 1.16385412687962, 0.05039533464000,
                   0.01757740533460],
        value_2 = [-0.38046562696294, -0.12266997083590, -0.99322817120634,
                   0.39034789080558, 1.16385412688026, 0.05039533464023,
                   0.01757740533476],
        value_3 = [-0.38046562696275, -0.12266997083484, -0.99322817120582,
                   0.39034789080555, 1.16385412688162, 0.05039533463988,
                   0.01757740533435]
        return [value_1, value_2, value_3]

    @staticmethod
    def define_players():
        player_vector_sizes = [3, 2, 2]
        player_objective_functions = [0, 1, 2]
        player_constraints = [[0, 1], [2], [3]]
        return [player_vector_sizes, player_objective_functions, player_constraints]

    @staticmethod
    def objective_functions():
        return [A3dev.obj_func_1, A3dev.obj_func_2, A3dev.obj_func_3]

    @staticmethod
    def objective_function_derivatives():
        return [A3dev.obj_func_der_1, A3dev.obj_func_der_2, A3dev.obj_func_der_3]

    @staticmethod
    def constraints():
        return [A3dev.g0, A3dev.g1, A3dev.g2, A3dev.g3, A3dev.g4, A3dev.g5]

    @staticmethod
    def constraint_derivatives():
        return [A3dev.g0_der, A3dev.g1_der, A3dev.g2_der, A3dev.g3_der, A3dev.g4_der, A3dev.g5_der]

    A1 = np.array([[20, 5, 3], [5, 5, -5], [3, -5, 15]])
    A2 = np.array([[11, -1], [-1, 9]])
    A3 = np.array([[48, 39], [39, 53]])
    B1 = np.array([[-6, 10, 11, 20], [10, -4, -17, 9], [15, 8, -22, 21]])
    B2 = np.array([[20, 1, -3, 12, 1], [10, -4, 8, 16, 21]])
    B3 = np.array([[10, -2, 22, 12, 16], [9, 19, 21, -4, 20]])
    b1 = np.array([[1], [-1], [1]])
    b2 = np.array([[1], [0]])
    b3 = np.array([[-1], [2]])

    # Define Functions below

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
    def obj_func_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        A1 = np.array([[20, 5, 3], [5, 5, -5], [3, -5, 15]])
        B1 = np.array([[-6, 10, 11, 20], [10, -4, -17, 9], [15, 8, -22, 21]])
        b1 = np.array([[1], [-1], [1]])
        x1 = x[0]
        x_n1 = np.vstack((x[1], x[2]))
        return A3dev.obj_func_der(x1, x_n1, A1, B1, b1)

    @staticmethod
    def obj_func_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        A2 = np.array([[11, -1], [-1, 9]])
        B2 = np.array([[20, 1, -3, 12, 1], [10, -4, 8, 16, 21]])
        b2 = np.array([[1], [0]])
        x2 = x[1]
        x_n2 = np.vstack((x[0], x[2]))
        return A3dev.obj_func_der(x2, x_n2, A2, B2, b2)

    @staticmethod
    def obj_func_3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        A3 = np.array([[48, 39], [39, 53]])
        B3 = np.array([[10, -2, 22, 12, 16], [9, 19, 21, -4, 20]])
        b3 = np.array([[-1], [2]])
        x3 = x[2]
        x_n3 = np.vstack((x[0], x[1]))
        return A3dev.obj_func_der(x3, x_n3, A3, B3, b3)

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
    def obj_func_der_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        A1 = np.array([[20, 5, 3], [5, 5, -5], [3, -5, 15]])
        B1 = np.array([[-6, 10, 11, 20], [10, -4, -17, 9], [15, 8, -22, 21]])
        b1 = np.array([[1], [-1], [1]])
        x1 = x[0].reshape(-1,1)
        x_n1 = np.vstack((x[1], x[2])).reshape(-1,1)
        return A3dev.obj_func_der(x1, x_n1, A1, B1, b1)

    @staticmethod
    def obj_func_der_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        A2 = np.array([[11, -1], [-1, 9]])
        B2 = np.array([[20, 1, -3, 12, 1], [10, -4, 8, 16, 21]])
        b2 = np.array([[1], [0]])
        x2 = x[1].reshape(-1,1)
        # print(np.vstack((x[0], x[2])))
        x_n2 = np.vstack((x[0].reshape(-1,1), x[2].reshape(-1,1)))
        return A3dev.obj_func_der(x2, x_n2, A2, B2, b2)

    @staticmethod
    def obj_func_der_3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        A_3 = np.array([[48, 39], [39, 53]])
        B3 = np.array([[10, -2, 22, 12, 16], [9, 19, 21, -4, 20]])
        b3 = np.array([[-1], [2]])
        x3 = x[2].reshape(-1,1)
        x_n3 = np.vstack((x[0].reshape(-1,1), x[1].reshape(-1,1)))
        return A3dev.obj_func_der(x3, x_n3, A_3, B3, b3)

    @staticmethod
    def g0(x):
        x1, x2, x3 = x
        return (np.sum(x1) + np.sum(x2) +np.sum(x3) - 20)

    @staticmethod
    def g1(x):
        x1, x2, x3 = x
        return (x1[0] + x1[1] - x1[2] - x2[0] + x3[1] - 5)

    @staticmethod
    def g2(x):
        x1, x2, x3 = x
        return (x2[0] + x2[1] - x1[1] - x1[2] + x3[0] - 7)

    @staticmethod
    def g3(x):
        x1, x2, x3 = x
        return (x3[1] - x1[0] - x1[2] + x2[0] - 4)

    @staticmethod
    def g4(x):
        x1, x2, x3 = x
        return -10 - np.vstack((x1.reshape(-1,1), x2.reshape(-1,1), x3.reshape(-1,1)))

    @staticmethod
    def g5(x):
        x1, x2, x3 = x
        return np.vstack((x1.reshape(-1,1), x2.reshape(-1,1), x3.reshape(-1,1))) - 10

    @staticmethod
    # partial g0 / partial x1
    def g0_der(x1):
        return np.array([[1, 1, 1]]).reshape(-1, 1)

    @staticmethod
    # partial g1 / partial x1
    def g1_der(x1):
        return np.array([[1, 1, -1]]).reshape(-1, 1)

    @staticmethod
    # partial g2 / partial x2
    def g2_der(x1):
        return np.array([1, 1]).reshape(-1, 1)

    @staticmethod
    # partial g3 / partial x3
    def g3_der(x1):
        return np.array([[ 0, 1]]).reshape(-1, 1)

    @staticmethod
    def g4_der(x1):
        return -1

    @staticmethod
    def g5_der(x1):
        return 1


def A3devrun(vars):
    players = construct_vectors(np.array(vars[:7]).reshape(-1,1), [3,2,2])
    raw_players = np.array(vars[:7]).reshape(-1,1)
    dual_player = np.array(vars[7:]).reshape(-1,1)
    constraints = A3dev.constraints()

    grad_obj_1 = A3dev.obj_func_der_1(players) # (3,1)
    grad_obj_2 = A3dev.obj_func_der_2(players) # (2,1)
    grad_obj_3 = A3dev.obj_func_der_3(players) # (2,1)
    grad_obj = np.vstack((grad_obj_1, grad_obj_2, grad_obj_3))

    grad_cons_1 = dual_player[0] * A3dev.g0_der(players) + dual_player[1] * A3dev.g1_der(players) # (3,1)
    grad_cons_2 = dual_player[2] * A3dev.g2_der(players)   # (2,1)
    grad_cons_3 = dual_player[3] * A3dev.g3_der(players)   # (2,1)
    grad_cons = np.vstack((grad_cons_1, grad_cons_2, grad_cons_3))
    grad_cons += dual_player[4] * A3dev.g4_der(players)
    grad_cons += dual_player[5] * A3dev.g5_der(players)
    grad = grad_obj + grad_cons

    eng = grad.T @ grad

    grad_dual = []
    for jdx, constraint in enumerate(constraints):
        g = -constraint(players)
        # g = np.where(
        #     g <= 0,
        #     g**2,
        #     g**2 * np.tanh(dual_constraints[jdx])
        # )
        g = (dual_player[jdx] ** 2 / (1 + dual_player[jdx] ** 2)) * (g ** 2 / (1 + g ** 2)) + np.exp(-dual_player[jdx] ** 2) * (
                    np.maximum(0, -g) ** 2 / (1 + np.maximum(0, -g) ** 2))
        grad_dual.append(g.flatten())
    g_dual = np.concatenate(grad_dual).reshape(-1, 1)

    return sum(eng) + sum(g_dual)

x1 = np.array([[0], [0], [0]])
x2 = np.array([[0], [0]])
x3 = np.array([[0], [0]])
x = np.array([x1, x2, x3], dtype=object)
d = np.array([1,1,1,1,1,1])
ip = flatten_variables(x, d).tolist()

optimize = True
if optimize:
    minimizer_kwargs = dict(method="L-BFGS-B")
    start = timeit.default_timer()
    res1=basinhopping(A3devrun, ip, stepsize=0.01, niter=1000, minimizer_kwargs=minimizer_kwargs, interval=1, niter_success=100, disp = True)
    stop = timeit.default_timer()

    print("Result: ", res1.x[:7])
    print("Time: ", stop - start)
    print("Constraints: ", res1.x[7:])

    print("Energy: ", A3devrun(res1.x))
