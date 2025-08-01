import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt
from library.misc import *

class A11dev:
    @staticmethod
    def paper_solution():
        value_1 = [0.75002417863494, 0.2500241786374]
        return [value_1]

    @staticmethod
    def define_players():
        player_vector_sizes = [1, 1]
        player_objective_functions = [0, 1]
        player_constraints = [[0], [0]]
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 100.0)]
        bounds_training = [(0.0, 1.0), (0.0, 1.0), (0.0, 100.0)]
        return [player_vector_sizes, player_objective_functions, player_constraints, bounds, bounds_training]

    @staticmethod
    def objective_functions():
        return [A11dev.obj_func_1, A11dev.obj_func_2]

    @staticmethod
    def objective_function_derivatives():
        return [A11dev.obj_func_der_1, A11dev.obj_func_der_2]

    @staticmethod
    def constraints():
        return [A11dev.g0]

    @staticmethod
    def constraint_derivatives():
        return [A11dev.g0_der]

    @staticmethod
    def obj_func_1(x: npt.NDArray[np.float64]) -> float:
        x1 = x[0]
        return (x1 - 1) ** 2

    @staticmethod
    def obj_func_2(x: npt.NDArray[np.float64]) -> float:
        x2 = x[1]
        return (x2 - 0.5) ** 2

    @staticmethod
    def obj_func_der_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        grad = 2 * (x[0] - 1)  # d/dx1 of (x1 - 1)^2
        return grad

    @staticmethod
    def obj_func_der_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        grad = 2 * (x[1] - 0.5)  # d/dx2 of (x2 - 0.5)^2
        return grad

    @staticmethod
    def g0(x: npt.NDArray[np.float64]) -> float:
        return x[0] + x[1] - 1

    @staticmethod
    def g0_der(x: npt.NDArray[np.float64]) -> float:
        return np.array([[1, 1]]).reshape(-1, 1)

def A11devrun(x):
    primal = np.array(x[:2]).reshape(-1,1)
    dual = np.array(x[2:]).reshape(-1, 1)  # lower and upper bound
    constraints = A11dev.constraints()

    grad1 = A11dev.obj_func_der_1(primal)
    grad2 = A11dev.obj_func_der_2(primal)
    grad = np.vstack((grad1, grad2)).reshape(-1,1)
    grad += A11dev.g0_der(primal) * dual[0]

    eng = grad.T @ grad

    # Dual player
    grad_dual = []
    for jdx, constraint in enumerate(constraints):
        g = -constraint(primal)
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
    primal = [0, 0]
    dual_ip = [1]
    ip1 = primal + dual_ip
    A11devrun(ip1)
    start = timeit.default_timer()
    res1 = basinhopping(A11devrun, ip1, stepsize=0.01, niter=1000, minimizer_kwargs=minimizer_kwargs, niter_success=100,
                        interval=1, disp=True)
    stop = timeit.default_timer()

    print('Time: ', stop - start)
    print(res1.x)