import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt


class A14dev:
    @staticmethod
    def paper_solution():
        value_1 = [ 0.08999991899425, 0.08999991899426, 0.08999991899425,
                    0.08999991899425, 0.08999991899425, 0.08999991899425,
                    0.08999991899425, 0.08999991899426, 0.08999991899425,
                    0.08999991899425]
        return [value_1]

    @staticmethod
    def define_players():
        B = 1
        player_vector_sizes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        player_objective_functions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        player_constraints = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
        return [player_vector_sizes, player_objective_functions, player_constraints]

    @staticmethod
    def objective_functions():
        return [A14dev.obj_func]

    @staticmethod
    def objective_function_derivatives():
        return [A14dev.obj_func_der]

    @staticmethod
    def constraints():
        return [A14dev.g0, A14dev.g1]

    @staticmethod
    def constraint_derivatives() -> List:
        return [A14dev.g0_der, A14dev.g1_der]

    @staticmethod
    def obj_func(x):
        # x: numpy array (N, 1)
        # B: constant
        S = sum(x)
        B = 1
        obj = (-x / S) * (1 - S / B)
        return obj

    @staticmethod
    def obj_func_der(x):
        # x: numpy array (N,1)
        # B: constant
        x = np.concatenate(x).reshape(-1, 1)
        B = 1
        S = sum(x)
        obj = ((x - S) / S) ** 2 + (1 / B)
        return obj

    @staticmethod
    def g0(x):
        B = 1
        return sum(x) - B

    @staticmethod
    def g1(x):
        x = np.concatenate(x).reshape(-1, 1)
        return 0.01 - x

    @staticmethod
    def g0_der(x):
        return 1

    @staticmethod
    def g1_der(x):
        return -1

def A14devrun(x):
    players = np.array(x[:10]).reshape(-1, 1)  # two player each with one variable
    dual = np.array(x[10:]).reshape(-1, 1)  # lower and upper bound
    constraints = A14dev.constraints()

    grad = A14dev.obj_func_der(players)
    grad += dual[0]
    grad -= dual[1]
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
    primal = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    dual_ip = [1, 1]
    ip1 = primal + dual_ip
    start = timeit.default_timer()
    res1 = basinhopping(A14devrun, ip1, stepsize=0.01, niter=1000, minimizer_kwargs=minimizer_kwargs, niter_success=100,
                        interval=1, disp=True)
    stop = timeit.default_timer()

    print('Time: ', stop - start)
    print(res1.x)