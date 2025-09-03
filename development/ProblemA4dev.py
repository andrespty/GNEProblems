import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt
from gne_solver.misc import *

class A4dev:
    @staticmethod
    def paper_solution():
        value_1 = [0.99982626069210, 0.99996267821636, 0.99987070414176,
                   0.99985869062731, 0.99983447394048, 0.99991824127925,
                   0.99991381820076]
        value_2 = [0.99982626069210, 0.99996267821636, 0.99987070414176,
                   0.99985869062731, 0.99983447394048, 0.99991824127925,
                   0.99991381820076]
        value_3 = [0.99991935573409, 0.99998267343165, 0.99993998316707,
                   0.99993440803353, 0.99992316858615, 0.99987984742324,
                   0.99987334759435]
        return [value_1, value_2, value_3]

    @staticmethod
    def define_players():
        player_vector_sizes = [3, 2, 2]
        player_objective_functions = [0, 1, 2]
        player_constraints = [[0, 1, 4, 5], [2, 4, 5], [3, 4, 5]]
        return [player_vector_sizes, player_objective_functions, player_constraints]

    @staticmethod
    def objective_functions():
        return [A4dev.obj_func_1, A4dev.obj_func_2, A4dev.obj_func_3]

    @staticmethod
    def objective_function_derivatives():
        return [A4dev.obj_func_der_1, A4dev.obj_func_der_2, A4dev.obj_func_der_3]

    @staticmethod
    def constraints():
        return [A4dev.g0, A4dev.g1, A4dev.g2, A4dev.g3, A4dev.g4, A4dev.g5]

    @staticmethod
    def constraint_derivatives():
        return [A4dev.g0_der, A4dev.g1_der, A4dev.g2_der, A4dev.g3_der, A4dev.g4_der, A4dev.g5_der]

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
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        A1 = np.array([
            [20 + (x2[0, 0] ** 2), 5, 3],
            [5, 5 + (x2[1, 0] ** 2), -5],
            [3, -5, 15]
        ])

        B1 = np.array([[-6, 10, 11, 20], [10, -4, -17, 9], [15, 8, -22, 21]])
        b1 = np.array([[1], [-1], [1]])
        x1 = x[0]
        x_n1 = np.vstack((x[1], x[2]))
        return A4dev.obj_func(x1, x_n1, A1, B1, b1)

    @staticmethod
    def obj_func_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        A2 = np.array([
            [11 + (x3[0, 0] ** 2), -1],
            [-1, 9]
        ])

        A2 = np.array([[11, -1], [-1, 9]])
        B2 = np.array([[20, 1, -3, 12, 1], [10, -4, 8, 16, 21]])
        b2 = np.array([[1], [0]])
        x_n2 = np.vstack((x1, x3))
        return A4dev.obj_func(x2, x_n2, A2, B2, b2)

    @staticmethod
    def obj_func_3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        A3 = np.array([
            [48, 39],
            [39, 53 + (x1[0, 0] ** 2)]
        ])

        B3 = np.array([[10, -2, 22, 12, 16], [9, 19, 21, -4, 20]])
        b3 = np.array([[-1], [2]])
        x_n3 = np.vstack((x1, x2))
        return A4dev.obj_func(x3, x_n3, A3, B3, b3)

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
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        A1 = np.array([
            [20 + (x2[0, 0] ** 2), 5, 3],
            [5, 5 + (x2[1, 0] ** 2), -5],
            [3, -5, 15]
        ])
        B1 = np.array([[-6, 10, 11, 20], [10, -4, -17, 9], [15, 8, -22, 21]])
        b1 = np.array([[1], [-1], [1]])
        x_n1 = np.vstack((x2, x3))
        return A4dev.obj_func_der(x1, x_n1, A1, B1, b1)

    @staticmethod
    def obj_func_der_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        A2 = np.array([
            [11 + (x3[0, 0] ** 2), -1],
            [-1, 9]
        ])
        B2 = np.array([[20, 1, -3, 12, 1], [10, -4, 8, 16, 21]])
        b2 = np.array([[1], [0]])
        x_n2 = np.vstack((x1, x3))
        return A4dev.obj_func_der(x2, x_n2, A2, B2, b2)

    @staticmethod
    def obj_func_der_3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        A3 = np.array([
            [48, 39],
            [39, 53 + (x1[0, 0] ** 2)]
        ])
        B3 = np.array([[10, -2, 22, 12, 16], [9, 19, 21, -4, 20]])
        b3 = np.array([[-1], [2]])
        x_n3 = np.vstack((x1, x2))
        return A4dev.obj_func_der(x3, x_n3, A3, B3, b3)

    @staticmethod
    def g0(x):
        x1, x2, x3 = x
        return (sum(x1) + sum(x2) + sum(x3) - 20)[0]

    @staticmethod
    def g1(x):
        x1, x2, x3 = x
        return (x1[0] + x1[1] - x1[2] - x2[0] + x3[1] - 5)[0]

    @staticmethod
    def g2(x):
        x1, x2, x3 = x
        return (x2[0] + x2[1] - x1[1] - x1[2] + x3[0] - 7)[0]

    @staticmethod
    def g3(x):
        x1, x2, x3 = x
        return (x3[1] - x1[0] - x1[2] + x2[0] - 4)[0]

    @staticmethod
    def g4(x):
        x1, x2, x3 = x
        return np.vstack((x1.reshape(-1,1), x2.reshape(-1,1), x3.reshape(-1,1))) - 10

    @staticmethod
    def g5(x):
        x1, x2, x3 = x
        return 1 - np.vstack((x1.reshape(-1, 1), x2.reshape(-1, 1), x3.reshape(-1, 1)))

    # partial g0 / partial x1
    @staticmethod
    def g0_der(x1):
        return np.array([[1, 1, 1, 1, 1, 1, 1]]).reshape(-1, 1)

    # partial g1 / partial x1
    @staticmethod
    def g1_der(x1):
        return np.array([[1, 1, -1, -1, 0, 0, 1]]).reshape(-1, 1)

    # partial g2 / partial x2
    @staticmethod
    def g2_der(x1):
        return np.array([[0, -1, -1, 1, 1, 1, 0]]).reshape(-1, 1)

    # partial g3 / partial x3
    @staticmethod
    def g3_der(x1):
        return np.array([[-1, 0, -1, 1, 0, 0, 1]]).reshape(-1, 1)

    @staticmethod
    def g4_der(x1):
        return 1

    @staticmethod
    def g5_der(x1):
        return -1



def A4devrun(vars):
    players = construct_vectors(np.array(vars[:7]).reshape(-1,1), [3,2,2])
    raw_players = np.array(vars[:7]).reshape(-1,1)
    dual_player = np.array(vars[7:]).reshape(-1,1)
    constraints = A4dev.constraints()
    constraints_der = A4dev.constraint_derivatives()
    player_constraints = A4dev.define_players()[2]
    constraints_mask = one_hot_encoding(player_constraints, [3,2,2], len(constraints))

    grad_obj_1 = A4dev.obj_func_der_1(players) # (3,1)
    grad_obj_2 = A4dev.obj_func_der_2(players) # (2,1)
    grad_obj_3 = A4dev.obj_func_der_3(players) # (2,1)
    grad_obj = np.vstack((grad_obj_1, grad_obj_2, grad_obj_3)).reshape(-1,1)
    for cdx, c_mask in enumerate(constraints_mask.T):
        grad_obj += c_mask.reshape(-1,1) * constraints_der[cdx](players) * dual_player[cdx]

    eng = grad_obj.T @ grad_obj

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

run = True
if run:
    #Cant solve it
    minimizer_kwargs = dict(method="L-BFGS-B")
    start = timeit.default_timer()
    res1=basinhopping(A4devrun, ip, stepsize=0.01, niter=1000, minimizer_kwargs=minimizer_kwargs, interval=1, niter_success=100, disp = True)
    stop = timeit.default_timer()

    print("Result: ", res1.x[:7])
    print("Time: ", stop - start)
    print("Constraints: ", res1.x[7:])

    print("Energy: ", A4devrun(res1.x))
