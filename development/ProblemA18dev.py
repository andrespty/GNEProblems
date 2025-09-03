import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt
from gne_solver.misc import *

class A18dev:
    @staticmethod
    def paper_solution():
        return []

    @staticmethod
    def define_players():
        player_vector_sizes = [6, 6]
        player_objective_functions = [0, 1]
        player_constraints = [[0,1,4,5,6, 7, 8, 9, 10],[2,3,4,5,6, 7, 8, 9, 10]]
        return [player_vector_sizes, player_objective_functions, player_constraints]
     
    @staticmethod
    def objective_functions():
        return [A18dev.obj_func_1, A18dev.obj_func_2]

    @staticmethod
    def objective_function_derivatives():
        return [A18dev.obj_func_der_1, A18dev.obj_func_der_2]
 
    @staticmethod
    def constraints():
        return [A18dev.g0, A18dev.g1, A18dev.g2, A18dev.g3, A18dev.g4, A18dev.g5, A18dev.g6, A18dev.g7, A18dev.g8, A18dev.g9, A18dev.g10]

    @staticmethod
    def constraint_derivatives():
        return [A18dev.g0_der, A18dev.g1_der, A18dev.g2_der, A18dev.g3_der, A18dev.g4_der, A18dev.g5_der, A18dev.g6_der, A18dev.g7_der, A18dev.g8_der, A18dev.g9_der, A18dev.g10_der]
   
    @staticmethod
    def abbreviations(x: npt.NDArray[np.float64]) -> float:
        x1 = x[0]
        x2 = x[1]
        S1 = 40 - (40/500)* (x1[0] + x1[3] + x2[0] + x2[3])
        S2 = 35 - (35/400)* (x1[1] + x1[4] + x2[1] + x2[4])
        S3 = 32 - (32/600)* (x1[2] + x1[5] + x2[2] + x2[5])
        return [S1, S2, S3]

    @staticmethod
    def obj_func_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        S1 = 40 - 40/500 * (x1[0] + x1[3] + x2[0] + x2[3])
        S2 = 35 - 35/400 * (x1[1] + x1[4] + x2[1] + x2[4])
        S3 = 32 - 32/600 * (x1[2] + x1[5] + x2[2] + x2[5])
        return (15 - S1)(x1[0] + x1[3]) + (15 - S2)(x1[1] + x1[4]) + (15 - S3)(x1[2] + x1[5])

    @staticmethod
    def obj_func_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        S1 = 40 - 40/500 * (x1[0] + x1[3] + x2[0] + x2[3])
        S2 = 35 - 35/400 * (x1[1] + x1[4] + x2[1] + x2[4])
        S3 = 32 - 32/600 * (x1[2] + x1[5] + x2[2] + x2[5])
        return (15 - S1)(x2[0] + x2[3]) + (15 - S2)(x2[1] + x2[4]) + (15 - S3)(x2[2] + x2[3])
   
    @staticmethod
    def obj_func_der_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        derx14 = -25 + (40/500) * (2*x1[0] + 2*x1[3] + x2[0] + x2[3])
        derx25 = -20 + (35/400) * (2*x1[1] + 2*x1[4] + x2[1] + x2[4])
        derx36 = -17 + (32/600) * (2*x1[2] + 2*x1[5] + x2[2] + x2[5])
        return np.array([derx14, derx25, derx36, derx14, derx25, derx36]).reshape(-1, 1)
   
   
    @staticmethod
    def obj_func_der_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        derx14 = -25 + (40 / 500) * (x1[0] + x1[3] + 2*x2[0] + 2*x2[3])
        derx25 = -20 + (35 / 400) * (x1[1] + x1[4] + 2*x2[1] + 2*x2[4])
        derx36 = -17 + (32 / 600) * (x1[2] + x1[5] + 2*x2[2] + 2*x2[5])
        return np.array([derx14, derx25, derx36, derx14, derx25, derx36]).reshape(-1, 1)

    @staticmethod
    def g0(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        # x1[0] + x1[1] + x1[2]
        return np.sum(x1[:3]) - 100
   
    @staticmethod
    def g1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        # x1[3] + x1[4] + x1[5]
        return np.sum(x1[3:]) - 50
   
    @staticmethod
    def g2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        # x2[0] + x2[1] + x2[2]
        return np.sum(x2[:3]) - 100
   
    @staticmethod
    def g3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        # x2[3] + x2[4] + x2[5]
        return x2[3:]- 50
   
    @staticmethod
    def g4(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        S1, S2, S3 = A18dev.abbreviations(x)
        return S1 - S2 - 1

    @staticmethod
    def g5(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        S1, S2, S3 = A18dev.abbreviations(x)
        return S2 - S1 - 1
   
    @staticmethod
    def g6(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        S1, S2, S3 = A18dev.abbreviations(x)
        return S1 - S3 - 1

    @staticmethod
    def g7(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        S1, S2, S3 = A18dev.abbreviations(x)
        return S3 - S1 - 1

    @staticmethod
    def g8(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        S1, S2, S3 = A18dev.abbreviations(x)
        return S2 - S3 - 1

    @staticmethod
    def g9(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        S1, S2, S3 = A18dev.abbreviations(x)
        return S3 - S2 - 1

    @staticmethod
    def g10(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x = np.concatenate(x).reshape(-1,1)
        return 0 - x
   
    @staticmethod
    def g0_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array ([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).reshape(-1, 1)
   
    @staticmethod
    def g1_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array ([[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]]).reshape(-1, 1)
   
    @staticmethod
    def g2_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array ([[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]]).reshape(-1, 1)
   
    @staticmethod
    def g3_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]).reshape(-1, 1)
   
    @staticmethod
    def g4_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array ([[-40/500, 35/400, 0, -40/500, 35/400, 0,
                           -40/500, 35/400, 0, -40/500, 35/400, 0]]).reshape(-1, 1)

    @staticmethod
    def g5_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([[40/500, -35/400, 0, 40/500, -35/400, 0,
                          40/500, -35/400, 0, 40/500, -35/400, 0]]).reshape(-1, 1)

    @staticmethod
    def g6_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array ([[-40/500, 0, 32/600, -40/500, 0, 32/600,
                           -40/500, 0, 32/600, -40/500, 0, 32/600]]).reshape(-1, 1)

    @staticmethod
    def g7_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([[40/500, 0, -32/600, 40/500, 0, -32/600,
                          40/500, 0, -32/600, 40/500, 0, -32/600]]).reshape(-1, 1)

    @staticmethod
    def g8_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array ([[0, -35/400, 32/600, 0, -35/400, 32/600,
                           0, -35/400, 32/600, 0, -35/400, 32/600]]). reshape(-1, 1)

    @staticmethod
    def g9_der(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([[0, 35/400, -32/600, 0, 35/400, -32/600,
                          0, 35/400, -32/600, 0, 35/400, -32/600]]).reshape(-1, 1)

    @staticmethod
    def g10_der(x):
        return -1
   
def A18devrun(x):
    primal = np.array(x[:12]).reshape(-1,1)
    players = construct_vectors(primal, [6, 6] )
    dual = np.array(x[12:]).reshape(-1, 1)  # lower and upper bound
    constraints = A18dev.constraints()

    grad1 = A18dev.obj_func_der_1(players)
    grad2 = A18dev.obj_func_der_2(players)
    grad = np.vstack((grad1, grad2)).reshape(-1,1)

    for cdx, constraint_der in enumerate(A18dev.constraint_derivatives()):
        grad += constraint_der(primal) * dual[cdx]

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
    primal = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    dual_ip = [1,1,1,1,1,1,1,1,1,1,1]
    ip1 = primal + dual_ip
    A18devrun(ip1)
    start = timeit.default_timer()
    res1 = basinhopping(A18devrun, ip1, stepsize=0.01, niter=1000, minimizer_kwargs=minimizer_kwargs, niter_success=100,
                        interval=1, disp=True)
    stop = timeit.default_timer()

    print('Time: ', stop - start)
    print(res1.x)
    print(A18devrun(res1.x))