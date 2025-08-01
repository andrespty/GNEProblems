import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt


class A12dev:
    @staticmethod
    def paper_solution():
        value_1 = [5.33331555561568, 5.33331555561]
        return [value_1]

    @staticmethod
    def define_players():
        player_vector_sizes = [1, 1]
        player_objective_functions = [0, 1]
        player_constraints = [[None], [None]]
        bounds = [(-10, 10), (-10, 10)]
        bounds_training = [(-10, 10), (-10, 10)]
        return [player_vector_sizes, player_objective_functions, player_constraints, bounds, bounds_training]

    @staticmethod
    def objective_functions():
        return [A12dev.obj_func_1, A12dev.obj_func_2]

    @staticmethod
    def objective_function_derivatives():
        return [A12dev.obj_func_der_1, A12dev.obj_func_der_2]

    @staticmethod
    def constraints():
        return [A12dev.g0, A12dev.g1]

    @staticmethod
    def constraint_derivatives():
        return [A12dev.g0_der, A12dev.g1_der]

    @staticmethod
    def obj_func_1(x: npt.NDArray[np.float64]) -> float:
        return x[0] * (x[0] + x[1] - 16)

    @staticmethod
    def obj_func_2(x: npt.NDArray[np.float64]) -> float:
        return x[1] * (x[0] + x[1] - 16)

    @staticmethod
    def obj_func_der_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        grad = 2 * x[0] + x[1] - 16
        return grad

    @staticmethod
    def obj_func_der_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        grad = x[0] + 2 * x[1] - 16
        return grad

    @staticmethod
    def g0(x):
        return -10 - x

    @staticmethod
    def g1(x):
        return x - 10

    @staticmethod
    def g0_der(x):
        return -1

    @staticmethod
    def g1_der(x):
        return 1

def A12devrun(x):
    players = np.array(x[:2]).reshape(-1, 1)  # two player each with one variable
    dual = np.array(x[2:]).reshape(-1, 1)  # lower and upper bound
    constraints = A12dev.constraints()

    # Primal player
    grad1 = A12dev.obj_func_der_1(x)
    grad2 = A12dev.obj_func_der_2(x)
    grad = np.vstack((grad1, grad2)).reshape(-1, 1)
    grad += dual[0]  # lower bound
    grad += dual[1]  # upper bound
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
    primal = [0, 0]
    dual_ip = [1, 1]
    ip1 = primal + dual_ip
    start = timeit.default_timer()
    res1 = basinhopping(A12devrun, ip1, stepsize=0.01, niter=1000, minimizer_kwargs=minimizer_kwargs, niter_success=100,
                        interval=1, disp=True)
    stop = timeit.default_timer()

    print('Time: ', stop - start)
    print(res1.x)