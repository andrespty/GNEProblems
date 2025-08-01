import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt


class A13dev:
    @staticmethod
    def paper_solution():
        value_1 = [21.14480155732168,
                   16.02785326538717,
                   2.7259709656438]
        return [value_1]

    @staticmethod
    def define_players():
        player_vector_sizes = [1, 1, 1]
        player_objective_functions = [0, 1, 2]
        player_constraints = [[0, 1], [0, 1], [0, 1]]
        return [player_vector_sizes, player_objective_functions, player_constraints]

    @staticmethod
    def objective_functions():
        return [A13dev.obj_func_1, A13dev.obj_func_2, A13dev.obj_func_3]

    @staticmethod
    def objective_function_derivatives():
        return [A13dev.obj_func_der_1, A13dev.obj_func_der_2, A13dev.obj_func_3]

    @staticmethod
    def constraints():
        return [A13dev.g0, A13dev.g1]

    @staticmethod
    def constraint_derivatives():
        return [A13dev.g0_der, A13dev.g1_der]

    @staticmethod
    def obj_func(
            x: npt.NDArray[np.float64],
            c1: npt.NDArray[np.float64],
            c2: npt.NDArray[np.float64],
            S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # x: actions vector (d_i, 1)
        # A: constant matrix (m, n)
        # B: constant matrix (m, sum(d_{-i}))
        # b: constant vector (m, 1)
        d1 = 3
        d2 = 0.01
        obj = x * (c1 + c2 * x - d1 + d2 * S)
        return obj

    @staticmethod
    def obj_func_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        S = x1 + x2 + x3
        c1 = 0.10
        c2 = 0.01
        return A13dev.obj_func(x1, c1, c2, S)

    @staticmethod
    def obj_func_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        S = x1 + x2 + x3
        c1 = 0.12
        c2 = 0.05
        return A13dev.obj_func(x2, c1, c2, S)

    @staticmethod
    def obj_func_3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        S = x1 + x2 + x3
        c1 = 0.15
        c2 = 0.01
        return A13dev.obj_func(x3, c1, c2, S)

    @staticmethod
    def obj_func_der(
            x: npt.NDArray[np.float64],
            c1: npt.NDArray[np.float64],
            c2: npt.NDArray[np.float64],
            S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # x: actions vector (d_i, 1)
        # A: constant matrix (m, n)
        # B: constant matrix (m, sum(d_{-i}))
        # b: constant vector (m, 1)
        d1 = 3
        d2 = 0.01
        obj = c1 + 2 * c2 * x - d1 + d2 * (S + x)
        return obj

    @staticmethod
    def obj_func_der_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        S = x1 + x2 + x3
        c1 = 0.10
        c2 = 0.01
        return A13dev.obj_func_der(x1, c1, c2, S)

    @staticmethod
    def obj_func_der_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        S = x1 + x2 + x3
        c1 = 0.12
        c2 = 0.05
        return A13dev.obj_func_der(x2, c1, c2, S)

    @staticmethod
    def obj_func_der_3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        S = x1 + x2 + x3
        c1 = 0.15
        c2 = 0.01
        return A13dev.obj_func_der(x3, c1, c2, S)

    @staticmethod
    def g0(x: npt.NDArray[np.float64]) -> float:
        return 3.25 * x[0] + 1.25 * x[1] + 4.125 * x[2] - 100

    @staticmethod
    def g1(x: npt.NDArray[np.float64]) -> float:
        return 2.2915 * x[0] + 1.5625 * x[1] + 2.814 * x[2] - 100

    @staticmethod
    def g0_der(x: npt.NDArray[np.float64]) -> float:
        return np.array([[3.25, 1.25, 4.125]]).reshape(-1, 1)

    @staticmethod
    def g1_der(x: npt.NDArray[np.float64]) -> float:
        return np.array([[2.2915, 1.5625, 2.814]]).reshape(-1,1)

def A13devrun(x):
    players = np.array(x[:3]).reshape(-1, 1)  # two player each with one variable
    dual = np.array(x[3:]).reshape(-1, 1)  # lower and upper bound
    constraints = A13dev.constraints()

    grad1 = A13dev.obj_func_der_1(players)
    grad2 = A13dev.obj_func_der_2(players)
    grad3 = A13dev.obj_func_der_3(players)
    grad = np.vstack((grad1, grad2, grad3)).reshape(-1,1)
    grad += A13dev.g0_der(players) * dual[0]
    grad += A13dev.g1_der(players) * dual[1]
    eng = grad.T @ grad

    # Dual player
    grad_dual = []
    for jdx, constraint in enumerate(constraints):
        g = -constraint(players)
        # g = np.where(

        #     g <= 0,
        #     g**2,
        #     g**2 * np.tanh(dual[jdx])
        # )
        g = (dual[jdx] ** 2 / (1 + dual[jdx] ** 2)) * (g ** 2 / (1 + g ** 2)) + np.exp(-dual[jdx] ** 2) * (
                np.maximum(0, -g) ** 2 / (1 + np.maximum(0, -g) ** 2))
        grad_dual.append(g.flatten())
    g_dual = np.concatenate(grad_dual).reshape(-1, 1)

    return eng + np.sum(g_dual)



run = True
if run:
    minimizer_kwargs = dict(method="L-BFGS-B")
    primal = [0, 0, 0]
    dual_ip = [1, 1]
    ip1 = primal + dual_ip
    start = timeit.default_timer()
    res1 = basinhopping(A13devrun, ip1, stepsize=0.01, niter=1000, minimizer_kwargs=minimizer_kwargs, niter_success=100,
                        interval=1, disp=True)
    stop = timeit.default_timer()

    print('Time: ', stop - start)
    print(res1.x)