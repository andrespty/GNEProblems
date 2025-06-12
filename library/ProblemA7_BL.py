import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt
from misc import *

class A7_BL:
    @staticmethod
    def paper_solution():
        value_1 = [
            0.99988245735506,
            0.99985542095046,
            0.99989138444537,
            0.99988866261891,
            0.99984494662577,
            0.99986703246906,
            0.99986897052169,
            0.99992059068103,
            0.99981225576918,
            1.00013812006334,
            0.99987211313045,
            1.84253230021096,
            0.99986555230493,
            0.99987070302597,
            0.99987574778109,
            0.99993185140789,
            0.99988068741824,
            0.99984157413000,
            0.99986193178624,
            0.99983143496263
        ]
        return [value_1]

    @staticmethod
    def define_players():
        player_vector_sizes = [5, 5, 5, 5]
        player_objective_functions = [0, 1, 2, 3]
        player_constraints = [[0],[1],[2], [3]]
        bounds = [(1, 5), (1, 5), (1, 5), (1, 5), (0, 100), (0, 100), (0, 100), (0, 100)]
        bounds_training = [(1, 5), (1, 5), (1, 5), (1, 5), (1, 5),
                           (1, 5), (1, 5), (1, 5), (1, 5), (1, 5),
                           (1, 5), (1, 5), (1, 5), (1, 5), (1, 5),
                           (1, 5), (1, 5), (1, 5), (1, 5), (1, 5),
                           (0, 100), (0, 100), (0, 100), (0, 100)]
        return [player_vector_sizes, player_objective_functions, player_constraints, bounds, bounds_training]

    @staticmethod
    def objective_functions():
        return [A7_BL.obj_func_1, A7_BL.obj_func_2, A7_BL.obj_func_3, A7_BL.obj_func_4]

    @staticmethod
    def objective_function_derivatives():
        return [A7_BL.obj_func_der_1, A7_BL.obj_func_der_2, A7_BL.obj_func_der_3,A7_BL.obj_func_der_4]

    @staticmethod
    def constraints():
        return [A7_BL.g0, A7_BL.g1, A7_BL.g2, A7_BL.g3]

    @staticmethod
    def constraint_derivatives():
        return [A7_BL.g0_der, A7_BL.g1_der, A7_BL.g2_der, A7_BL.g3_der]


    # Define everything below this comment
    @staticmethod
    def matrix():
        return np.array([
        [110, -3, 22, -14, -27, 1, 9, 19, -2, 23, -7, -20, -4, 22, -19, 22, 3, 13, -12, 18],
        [-3, 79, -9, -21, 18, 61, 0, 14, 58, -11, 4, -16, 20, -19, 13, -17, -1, 24, 22, 5],
        [22, -9, 90, 28, 22, -9, -21, -1, -5, 29, 15, -7, 4, 30, 2, 9, -1, -19, -60, 4],
        [-14, -21, 28, 106, 11, -33, -42, 14, 28, -10, 3, 6, 13, 22, -8, 6, -3, 15, -3, 0],
        [-27, 18, 22, 11, 134, 4, -4, -29, 39, -62, 74, 2, 4, -34, -1, 13, 8, 18, 12, 35],
        [1, 61, -9, -33, 4, 119, -14, 12, 12, -6, -23, -14, 16, -4, 15, -2, 8, 16, 9, -9],
        [9, 0, -21, -42, -4, -14, 72, -14, 6, -9, 12, 2, -24, 13, 29, 17, 13, -1, 19, 21],
        [19, 14, -1, 14, -29, 12, -14, 92, -10, 5, 8, 0, -4, 23, 8, -50, -11, 48, -8, 3],
        [-2, 58, -5, 28, 39, 12, 6, -10, 124, -39, -4, -16, 24, -18, 26, 4, 13, 29, 43, 23],
        [23, -11, 29, -10, -62, -6, -9, 5, -39, 130, -42, -21, 21, 68, -24, -21, -30, -54, -23, 9],
        [-7, 4, 15, 3, 74, -23, 12, 8, -4, -42, 138, -4, -24, -12, -27, 24, 21, 2, -10, 18],
        [-20, -16, -7, 6, 2, -14, 2, 0, -16, -21, -4, 89, -11, -14, -16, -32, -7, -5, 13, -4],
        [-4, 20, 4, 13, 4, 16, -24, -4, 24, 21, -24, -11, 107, 31, -3, -2, -22, 17, 4, 22],
        [22, -19, 30, 22, -34, -4, 13, 23, -18, 68, -12, -14, 31, 116, -1, 5, -18, -16, -43, 27],
        [-19, 13, 2, -8, -1, 15, 29, 8, 26, -24, -27, -16, -3, -1, 98, -4, -2, 50, 23, 8],
        [22, -17, 9, 6, 13, -2, 17, -50, 4, -21, 24, -32, -2, 5, -4, 102, 46, -29, -17, -1],
        [3, -1, -1, -3, 8, 8, 13, -11, 13, -30, 21, -7, -22, -18, -2, 46, 110, -16, 24, 12],
        [13, 24, -19, 15, 18, 16, -1, 48, 29, -54, 2, -5, 17, -16, 50, -29, -16, 102, 45, 14],
        [-12, 22, -60, -3, 12, 9, 19, -8, 43, -23, -10, 13, 4, -43, 23, -17, 24, 45, 119, 21],
        [18, 5, 4, 0, 35, -9, 21, 3, 23, 9, 18, -4, 22, 27, 8, -1, 12, 14, 21, 59]
    ])

    @staticmethod
    def obj_func(
            x: npt.NDArray[np.float64],
            x_ni: npt.NDArray[np.float64],
            A: npt.NDArray[np.float64],
            B: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # x: actions vector (d_i, 1)
        # A: constant matrix (m, n)
        # B: constant matrix (m, sum(d_{-i}))

        obj = 0.5 * x.T @ A @ x + x.T @ (B @ x_ni)
        return obj

    def obj_func_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        A1 = A7_BL.matrix()[0:5, 0:5]
        B1 = A7_BL.matrix()[0:5, 5:20]
        x_n1 = np.vstack((x2, x3, x4))
        return A7_BL.obj_func(x1, x_n1, A1, B1)

    @staticmethod
    def obj_func_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        A2 = A7_BL.matrix()[5:10, 5:10] #[x,y]
        B2 = np.hstack((A7_BL.matrix()[5:10, 0:5], A7_BL.matrix()[5:10, 10:20]))

        x_n2 = np.vstack((x1, x3, x4))
        return A7_BL.obj_func(x2, x_n2, A2, B2)

    @staticmethod
    def obj_func_3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        A3 = A7_BL.matrix()[10:15, 10:15]
        B3 = np.hstack((A7_BL.matrix()[10:15, 0:10], A7_BL.matrix()[10:15, 15:20]))
        x_n3 = np.vstack((x1, x2, x4))
        return A7_BL.obj_func(x3, x_n3, A3, B3)

    def obj_func_4(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        A4 = A7_BL.matrix()[15:20, 15:20]
        B4 = A7_BL.matrix()[15:20, 0:15]
        x_n4 = np.vstack((x1, x2, x3))
        return A7_BL.obj_func(x4, x_n4, A4, B4)

    @staticmethod
    def obj_func_der(
            x: npt.NDArray[np.float64],
            x_ni: npt.NDArray[np.float64],
            A: npt.NDArray[np.float64],
            B: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # x: actions vector (d_i, 1)
        # A: constant matrix (m, n)
        # B: constant matrix (m, sum(d_{-i}))
        obj = A @ x + B @ x_ni
        return obj

    @staticmethod
    def obj_func_der_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        A1 = A7_BL.matrix()[0:5, 0:5]
        B1 = A7_BL.matrix()[0:5, 5:20]

        x_n1 = np.vstack((x2, x3, x4))
        return A7_BL.obj_func_der(x1, x_n1, A1, B1)

    @staticmethod
    def obj_func_der_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        A2 = A7_BL.matrix()[5:10, 5:10]
        B2 = np.hstack((A7_BL.matrix()[5:10, 0:5], A7_BL.matrix()[5:10, 10:20]))
        x_n2 = np.vstack((x1, x3, x4))
        return A7_BL.obj_func_der(x2, x_n2, A2, B2)

    @staticmethod
    def obj_func_der_3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        A3 = A7_BL.matrix()[10:15, 10:15]
        B3 = np.hstack((A7_BL.matrix()[10:15, 0:10], A7_BL.matrix()[10:15, 15:20]))

        x_n3 = np.vstack((x1, x2, x4))
        return A7_BL.obj_func_der(x3, x_n3, A3, B3)

    @staticmethod
    def obj_func_der_4(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        A4 = A7_BL.matrix()[15:20, 15:20]
        B4 = A7_BL.matrix()[15:20, 0:15]

        x_n4 = np.vstack((x1, x2, x3))
        return A7_BL.obj_func_der(x4, x_n4, A4, B4)

    @staticmethod
    def g0(x):
        x1, x2, x3, x4 = x
        return (x1[0] + 2 * x1[1] - x1[2] + 3 * x1[3] - 4 * x1[4] - 2 + x2[1] - 3 * x2[2])[0]

    @staticmethod
    def g1(x):
        x1, x2, x3, x4 = x
        return (-x2[1] + 3 * x2[1] - 2 * x2[2] + x2[3] + 3 * x2[4] - 4 + x3[0] - 3 * x3[4] + 2 * x4[2])[0]

    @staticmethod
    def g2(x):
        x1, x2, x3, x4 = x
        return (-2 * x3[0] + 3 * x3[1] + x3[2] - x3[3] - 2 * x3[4] - 4 + x1[0] - 4 * x4[4])[0]

    @staticmethod
    def g3(x):
        x1, x2, x3, x4 = x
        return (4 * x4[0] - 2 * x4[1] - 3 * x4[2] - 6 * x4[3] + 5 * x4[4] - 3 + x1[0] + x1[1] - x2[0] - x2[1])[0]

    @staticmethod
    # partial g0 / partial x1
    def g0_der(x1):
        return np.array([[1, 2, -1, 3, -4]]).reshape(-1, 1)

    @staticmethod
    # partial g1 / partial x1
    def g1_der(x1):
        return np.array([[-1, 3, -2, 1, 3]]).reshape(-1, 1)

    @staticmethod
    # partial g2 / partial x2
    def g2_der(x1):
        return np.array([[-2, 3, 1, -1, -2]]).reshape(-1, 1)

    @staticmethod
    # partial g3 / partial x3
    def g3_der(x1):
        return np.array([[4, -2, -3, -6, 5]]).reshape(-1, 1)

# def get_engval(myvar, gradval, mylb, myub):
#   if gradval<=0:
#     engval=(myub-myvar)*np.log(1-gradval)
#   else:
#     engval=(myvar-mylb)*np.log(1+gradval)
#   return engval
def get_engval(myvar, gradval, mylb, myub):
    myvar = np.asarray(myvar)
    gradval = np.asarray(gradval)

    engval = np.empty_like(myvar, dtype=np.float64)

    mask = gradval <= 0
    engval[mask] = (myub - myvar[mask]) * np.log(1 - gradval[mask])
    engval[~mask] = (myvar[~mask] - mylb) * np.log(1 + gradval[~mask])

    return engval

def A7_ex(vars):
    primal_vars = 20
    action_sizes = [5,5,5,5]
    players = [vec.reshape(-1,1)  for vec in construct_vectors(vars[:primal_vars], action_sizes)]
    dual = vars[primal_vars:]

    # Players
    cost_p1 = A7_BL.obj_func_der_1(players) + dual[0] * A7_BL.g0_der(dual)
    cost_p2 = A7_BL.obj_func_der_2(players) + dual[1] * A7_BL.g1_der(dual)
    cost_p3 = A7_BL.obj_func_der_3(players) + dual[2] * A7_BL.g2_der(dual)
    cost_p4 = A7_BL.obj_func_der_4(players) + dual[3] * A7_BL.g3_der(dual)
    # p1_eng = np.linalg.norm(cost_p1) ** 2
    # p2_eng = np.linalg.norm(cost_p2) ** 2
    # p3_eng = np.linalg.norm(cost_p3) ** 2
    # p4_eng = np.linalg.norm(cost_p4) ** 2
    p1_eng = sum(get_engval(players[0], cost_p1, 1, 5))
    p2_eng = sum(get_engval(players[1], cost_p2, 1, 5))
    p3_eng = sum(get_engval(players[2], cost_p3, 1, 5))
    p4_eng = sum(get_engval(players[3], cost_p4, 1, 5))
    # Dual
    cost_d1 = -A7_BL.g0(players)
    cost_d2 = -A7_BL.g1(players)
    cost_d3 = -A7_BL.g2(players)
    cost_d4 = -A7_BL.g3(players)
    # d1_eng = np.linalg.norm(cost_d1) ** 2
    # d2_eng = np.linalg.norm(cost_d2) ** 2
    # d3_eng = np.linalg.norm(cost_d3) ** 2
    # d4_eng = np.linalg.norm(cost_d4) ** 2
    d1_eng = get_engval(dual[0], cost_d1, 0, 100)
    d2_eng = get_engval(dual[1], cost_d2, 0, 100)
    d3_eng = get_engval(dual[2], cost_d3, 0, 100)
    d4_eng = get_engval(dual[3], cost_d4, 0, 100)

    return p1_eng + p2_eng + p3_eng + p4_eng + d1_eng + d2_eng + d3_eng + d4_eng

def A7_sig(vars):
    primal_vars = 20
    action_sizes = [5,5,5,5]
    players_vars = [vec.reshape(-1,1)  for vec in construct_vectors(vars[:primal_vars], action_sizes)]
    dual_vars = vars[primal_vars:]
    bounds_primal = [vec.reshape(-1,2) for vec in construct_vectors(repeat_items([(1,5)],[primal_vars]), action_sizes)]
    players = [vectorized_sigmoid(vec, bounds_primal[i])  for i, vec in enumerate(players_vars)]
    dual = [vectorized_sigmoid(vec, np.array([0,100]).reshape(-1,2))  for i, vec in enumerate(dual_vars)]
    #
    # print(players)
    # print(dual)
    # Players
    cost_p1 = (A7_BL.obj_func_der_1(players) + dual[0] * A7_BL.g0_der(dual)) * (players[0] * (1-players[0]))
    cost_p2 = A7_BL.obj_func_der_2(players) + dual[1] * A7_BL.g1_der(dual) * (players[1] * (1-players[1]))
    cost_p3 = A7_BL.obj_func_der_3(players) + dual[2] * A7_BL.g2_der(dual) * (players[2] * (1-players[2]))
    cost_p4 = A7_BL.obj_func_der_4(players) + dual[3] * A7_BL.g3_der(dual) * (players[3] * (1-players[3]))
    p1_eng = np.linalg.norm(cost_p1) ** 2
    p2_eng = np.linalg.norm(cost_p2) ** 2
    p3_eng = np.linalg.norm(cost_p3) ** 2
    p4_eng = np.linalg.norm(cost_p4) ** 2

    # Dual
    cost_d1 = -A7_BL.g0(players) * (dual[0] * (1-dual[0]))
    cost_d2 = -A7_BL.g1(players) * (dual[1] * (1-dual[1]))
    cost_d3 = -A7_BL.g2(players) * (dual[2] * (1-dual[2]))
    cost_d4 = -A7_BL.g3(players) * (dual[3] * (1-dual[3]))
    d1_eng = np.linalg.norm(cost_d1) ** 2
    d2_eng = np.linalg.norm(cost_d2) ** 2
    d3_eng = np.linalg.norm(cost_d3) ** 2
    d4_eng = np.linalg.norm(cost_d4) ** 2

    return p1_eng + p2_eng + p3_eng + p4_eng + d1_eng + d2_eng + d3_eng + d4_eng

#
# bounds = Bounds([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0], [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,100,100,100,100])
# minimizer_kwargs = dict(method="SLSQP")
# ip1=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,10,10,10]
# start = timeit.default_timer()
# res1=basinhopping(A7_sig, ip1, stepsize=0.01, niter=1000, minimizer_kwargs=minimizer_kwargs, niter_success=100, interval=1, disp = True)
# stop = timeit.default_timer()
#
# print('Time: ', stop - start)
# print(res1.x)
# res_bounds = repeat_items([(1,5), (0,100)], [20,4])
# scaled_res = vectorized_sigmoid(np.array(res1.x).reshape(-1,1), np.array(res_bounds).reshape(-1, 2))
# print(scaled_res)
# print(A7_sig(scaled_res.tolist()))