import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt

from gne_solver.misc import construct_vectors


class A17dev:
    @staticmethod
    def paper_solution():
        value_1 = [0.00000737,
                   11.00002440,
                   7.99997560]
        return [value_1]

    @staticmethod
    def define_players():
        player_vector_sizes = [2, 1]
        player_objective_functions = [0, 1]
        player_constraints = [[0, 1], [0, 1]]
        return [player_vector_sizes, player_objective_functions, player_constraints]

    @staticmethod
    def objective_functions():
        return [A17dev.obj_func_1, A17dev.obj_func_2]

    @staticmethod
    def objective_function_derivatives():
        return [A17dev.obj_func_der_1, A17dev.obj_func_der_2]

    @staticmethod
    def constraints():
        return [A17dev.g0, A17dev.g1, A17dev.g2]

    @staticmethod
    def constraint_derivatives():
        return [A17dev.g0_der, A17dev.g1_der, A17dev.g2_der]

    @staticmethod
    def obj_func_1(x: npt.NDArray[np.float64]):
        x1 = x[0] # (2,1)
        x2 = x[1]
        return x1[0] ** 2 + x1[0] * x1[1] + x2 ** 2 + np.sum(x1) * x2 - 25 * x1[0] - 38 * x1[1]

    @staticmethod
    def obj_func_2(x: npt.NDArray[np.float64]):
        x1 = x[0] # (2,1)
        x2 = x[1]
        return x2 ** 2 + np.sum(x1) * x2 - 25 * x2

    @staticmethod
    def obj_func_der_1(x: npt.NDArray[np.float64]):
        x1 = x[0]
        x2 = x[1]
        der1 = 2 * x1[0] + x1[1] + x2 - 25
        der2 = x1[0] + 2 * x1[1] + x2 - 38
        return np.array([der1, der2]).reshape(-1,1)

    @staticmethod
    def obj_func_der_2(x: npt.NDArray[np.float64]):
        x1 = x[0]
        x2 = x[1]
        return 2 * x2 + np.sum(x1) - 25

    @staticmethod
    def g0(x: npt.NDArray[np.float64]):
        x1 = x[0]
        x2 = x[1]
        return x1[0] + 2 * x1[1] - x2 - 14

    @staticmethod
    def g1(x: npt.NDArray[np.float64]):
        x1 = x[0]
        x2 = x[1]
        return 3 * x1[0] + 2 * x1[1] + x2 - 30

    @staticmethod
    def g2(x: npt.NDArray[np.float64]):
        x = np.concatenate(x).reshape(-1,1)
        return 0 - x

    @staticmethod
    def g0_der(x: npt.NDArray[np.float64]):
        return np.array([[1, 2, -1]]).reshape(-1, 1)

    @staticmethod
    def g1_der(x: npt.NDArray[np.float64]):
        return np.array([[3, 2, 1]]).reshape(-1,1)

    @staticmethod
    def g2_der(x: npt.NDArray[np.float64]):
        return -1

def A17devrun(x):
    primal = np.array(x[:3]).reshape(-1,1)
    players = construct_vectors(primal, [2, 1] )
    dual = np.array(x[3:]).reshape(-1, 1)  # lower and upper bound
    constraints = A17dev.constraints()

    grad1 = A17dev.obj_func_der_1(players)
    grad2 = A17dev.obj_func_der_2(players)
    grad = np.vstack((grad1, grad2)).reshape(-1,1)
    grad += A17dev.g0_der(primal) * dual[0]
    grad += A17dev.g1_der(primal) * dual[1]
    grad += A17dev.g2_der(primal) * dual[2]

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
    dual_ip = [1,1,1]
    ip1 = primal + dual_ip
    A17devrun(ip1)
    start = timeit.default_timer()
    res1 = basinhopping(A17devrun, ip1, stepsize=0.01, niter=1000, minimizer_kwargs=minimizer_kwargs, niter_success=100,
                        interval=1, disp=True)
    stop = timeit.default_timer()

    print('Time: ', stop - start)
    print(res1.x)