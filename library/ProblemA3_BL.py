import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt

from library.misc import *


class A3_BL:

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
        bounds = [(-10, 10), (-10, 10), (-10, 10), (0, 100), (0, 100), (0, 100), (0, 100)]
        bounds_training = [(-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (0, 100),
                           (0, 100), (0, 100), (0, 100)]
        return [player_vector_sizes, player_objective_functions, player_constraints, bounds, bounds_training]

    @staticmethod
    def objective_functions():
        return [A3_BL.obj_func_1, A3_BL.obj_func_2, A3_BL.obj_func_3]

    @staticmethod
    def objective_function_derivatives():
        return [A3_BL.obj_func_der_1, A3_BL.obj_func_der_2, A3_BL.obj_func_der_3]

    @staticmethod
    def constraints():
        return [A3_BL.g0, A3_BL.g1, A3_BL.g2, A3_BL.g3]

    @staticmethod
    def constraint_derivatives():
        return [A3_BL.g0_der, A3_BL.g1_der, A3_BL.g2_der, A3_BL.g3_der]

    A1 = np.array([[20, 5, 3], [5, 5, -5], [3, -5, 15]])
    A2 = np.array([[11, -1], [-1, 9]])
    A3_BL = np.array([[48, 39], [39, 53]])
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
        return A3_BL.obj_func_der(x1, x_n1, A1, B1, b1)

    @staticmethod
    def obj_func_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        A2 = np.array([[11, -1], [-1, 9]])
        B2 = np.array([[20, 1, -3, 12, 1], [10, -4, 8, 16, 21]])
        b2 = np.array([[1], [0]])
        x2 = x[1]
        x_n2 = np.vstack((x[0], x[2]))
        return A3_BL.obj_func_der(x2, x_n2, A2, B2, b2)

    @staticmethod
    def obj_func_3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        A_3 = np.array([[48, 39], [39, 53]])
        B3 = np.array([[10, -2, 22, 12, 16], [9, 19, 21, -4, 20]])
        b3 = np.array([[-1], [2]])
        x3 = x[2]
        x_n3 = np.vstack((x[0], x[1]))
        return A3_BL.obj_func_der(x3, x_n3, A_3, B3, b3)

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
        return A3_BL.obj_func_der(x1, x_n1, A1, B1, b1)

    @staticmethod
    def obj_func_der_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        A2 = np.array([[11, -1], [-1, 9]])
        B2 = np.array([[20, 1, -3, 12, 1], [10, -4, 8, 16, 21]])
        b2 = np.array([[1], [0]])
        x2 = x[1].reshape(-1,1)
        # print(np.vstack((x[0], x[2])))
        x_n2 = np.vstack((x[0].reshape(-1,1), x[2].reshape(-1,1)))
        return A3_BL.obj_func_der(x2, x_n2, A2, B2, b2)

    @staticmethod
    def obj_func_der_3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        A_3 = np.array([[48, 39], [39, 53]])
        B3 = np.array([[10, -2, 22, 12, 16], [9, 19, 21, -4, 20]])
        b3 = np.array([[-1], [2]])
        x3 = x[2].reshape(-1,1)
        x_n3 = np.vstack((x[0].reshape(-1,1), x[1].reshape(-1,1)))
        return A3_BL.obj_func_der(x3, x_n3, A_3, B3, b3)

    @staticmethod
    def g0(x):
        x1, x2, x3 = x
        return (sum(x1) + sum(x2) + sum(x3) - 20)

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
        return np.array([[1, 1]]).reshape(-1, 1)

    @staticmethod
    # partial g3 / partial x3
    def g3_der(x1):
        return np.array([[0, 1]]).reshape(-1, 1)

def get_engval(myvar, gradval, mylb, myub):
    if gradval<=0:
        engval=(myub-myvar)*np.log(1-gradval)
    else:
        engval=(myvar-mylb)*np.log(1+gradval)
    return engval

def A3_ex(vars):
    players = construct_vectors(np.array(vars[:7]), [3,2,2])
    dual = np.array(vars[7:])
    lb = np.array([-10,-10,-10, -10,-10,-10, -10]).reshape(-1,1)
    ub = np.array([10,10,10, 10,10,10, 10]).reshape(-1,1)
    # print(players)
    grad_obj_1 = A3_BL.obj_func_der_1(players) # (3,1)
    grad_obj_2 = A3_BL.obj_func_der_2(players) # (2,1)
    grad_obj_3 = A3_BL.obj_func_der_3(players) # (2,1)
    grad_obj = np.vstack((grad_obj_1, grad_obj_2, grad_obj_3))
    # print(grad_obj)

    grad_cons_1 = dual[0] * A3_BL.g0_der(players) + dual[1] * A3_BL.g1_der(players) # (3,1)
    grad_cons_2 = dual[2] * A3_BL.g2_der(players)   # (2,1)
    grad_cons_3 = dual[3] * A3_BL.g3_der(players)   # (2,1)
    grad_cons = np.vstack((grad_cons_1, grad_cons_2, grad_cons_3))
    grad = grad_obj + grad_cons
    # print("Grad: ", grad)
    # eng = np.where(
    #     grad <= 0,
    #     (ub - np.array(vars[:7]).reshape(-1,1)) * np.log(1 - grad),
    #     (np.array(vars[:7]).reshape(-1,1) - lb) * np.log(1 + grad)
    # )
    # print("Eng: ", eng)
    eng = ((ub - np.array(vars[:7]).reshape(-1,1))**2)  * ((np.array(vars[:7]).reshape(-1,1) - lb)**2) * grad**2
    grad_cons_1 = -A3_BL.g0(players)
    grad_cons_2 = -A3_BL.g1(players)
    grad_cons_3 = -A3_BL.g2(players)
    grad_cons_4 = -A3_BL.g3(players)
    grad_cons = np.vstack((grad_cons_1, grad_cons_2, grad_cons_3, grad_cons_4))
    # eng_cons = [get_engval(dplayer, grad_cons[i],0,100) for i, dplayer in enumerate(dual)]
    eng_cons = [ ((100-dplayer)**2) * ((dplayer)**2) * grad_cons[i]**2 for i, dplayer in enumerate(dual)]
    return sum(eng)[0] + sum(eng_cons)[0]

def A3_sig(vars):
    raw_players = np.array(vars[:7]).reshape(-1,1)
    dual = np.array(vars[7:]).reshape(-1,1)
    lb = np.array([-10,-10,-10, -10,-10,-10, -10]).reshape(-1,1)
    ub = np.array([10,10,10, 10,10,10, 10]).reshape(-1,1)

    # Transformed actions
    x = np.clip(raw_players, -500, 500)
    xd = np.clip(dual, -500, 500)
    players = construct_vectors((ub - lb) * (1/(1+np.exp(-x))) + lb, [3,2,2])    # (10,1)
    dual = 100/(1+np.exp(-xd))

    # print(players)
    grad_obj_1 = A3_BL.obj_func_der_1(players) # (3,1)
    grad_obj_2 = A3_BL.obj_func_der_2(players) # (2,1)
    grad_obj_3 = A3_BL.obj_func_der_3(players) # (2,1)
    grad_obj = np.vstack((grad_obj_1, grad_obj_2, grad_obj_3))
    # print(grad_obj)

    grad_cons_1 = dual[0] * A3_BL.g0_der(players) + dual[1] * A3_BL.g1_der(players) # (3,1)
    grad_cons_2 = dual[2] * A3_BL.g2_der(players)   # (2,1)
    grad_cons_3 = dual[3] * A3_BL.g3_der(players)   # (2,1)
    grad_cons = np.vstack((grad_cons_1, grad_cons_2, grad_cons_3))
    grad = grad_obj + grad_cons
    # print("Grad: ", grad)
    eng = np.abs(grad)**2
    # print("Eng: ", eng)

    grad_cons_1 = -A3_BL.g0(players)
    grad_cons_2 = -A3_BL.g1(players)
    grad_cons_3 = -A3_BL.g2(players)
    grad_cons_4 = -A3_BL.g3(players)
    grad_cons = np.vstack((grad_cons_1, grad_cons_2, grad_cons_3, grad_cons_4))
    eng_cons = np.abs(grad_cons)**2
    return sum(eng) + sum(eng_cons)

x1 = np.array([[0], [0], [0]])
x2 = np.array([[0], [0]])
x3 = np.array([[0], [0]])
x = np.array([x1, x2, x3], dtype=object)
d = np.array([0,0,0,0])
ip = flatten_variables(x, d)
print("Energy: ",A3_sig(ip))

isTransform = False

if isTransform:
    minimizer_kwargs = dict(method="L-BFGS-B")
    start = timeit.default_timer()
    res1 = basinhopping(A3_sig, ip, stepsize=0.0001, niter=1000, minimizer_kwargs=minimizer_kwargs, interval=1 ,
                        niter_success=100, disp=True)
    stop = timeit.default_timer()
    lb_t = np.array([-10,-10,-10, -10,-10,-10, -10]).reshape(-1,1)
    ub_t = np.array([10,10,10, 10,10,10, 10]).reshape(-1,1)
    x_t = np.clip(np.array(res1.x[:7]).reshape(-1, 1), -500, 500)
    xd_t = np.clip(res1.x[7:], -500, 500)
    actions_primal = (ub_t - lb_t) * (1 / (1 + np.exp(-x_t))) + lb_t
    actions_dual = 100 / (1 + np.exp(-xd_t))
else:
    bounds = Bounds([-10,-10,-10, -10,-10,-10, -10, 0,0,0,0], [10,10,10, 10,10,10, 10,100,100,100,100])
    minimizer_kwargs = dict(method="SLSQP", bounds=bounds)
    start = timeit.default_timer()
    res1=basinhopping(A3_ex, ip, stepsize=0.01, niter=1000, minimizer_kwargs=minimizer_kwargs, interval=1, niter_success=100, disp = True)
    stop = timeit.default_timer()


print("Result: ", res1.x)
if isTransform and actions_primal.any() and actions_dual.any():
    print('Primal Translated: ', actions_primal)
    print('Dual Translated: ', actions_dual)
    # print(sum(actions_primal))
    # print(A2_BL.g0(actions_primal))
    # print(A2_BL.g1(actions_primal))
print("Energy: ", A3_ex(res1.x) if isTransform else A3_ex(res1.x))
# print("Gradients: ", get_grad(res1.x, translate=isTransform))
