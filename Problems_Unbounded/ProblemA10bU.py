import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt

from Problems_Bounded.ProblemA10a import A10a
from library.misc import construct_vectors


class A10b:
    F= 4
    C= 20
    P= 5
    N = F + C + 1

    @staticmethod
    def define_players():
        player_vector_sizes = [A10b.P for _ in range(A10b.N)]
        player_objective_functions = [0 for _ in range(A10b.N)]  # change to all 0s

        f_player_constraints = [[0] for _ in range(A10b.F)]
        c_player_constraints = [[1] for _ in range(A10b.C)]
        player_constraints = f_player_constraints + c_player_constraints + [[2,3]]
        return [player_vector_sizes, player_objective_functions, player_constraints]

    @staticmethod
    def objective_functions():
        return [A10b.obj_func_firms,
                A10b.obj_func_consumers_1,
                A10b.obj_func_market
        ]

    @staticmethod
    def objective_function_derivatives():
        return [A10b.obj_der]

    @staticmethod
    def constraints():
        return [A10b.g0, A10b.g1, A10b.g2, A10b.g3, A10b.g4]

    @staticmethod
    def constraint_derivatives():
        return [A10b.g0_der, A10b.g1_der, A10b.g2_der, A10b.g3_der, A10b.g4_der]

    @staticmethod
    def obj_func_firms(x):
        p = x[-1].reshape(-1,1)
        y = np.hstack(x[:A10b.F]).reshape(A10b.P, -1)
        return p.T @ y # returns (1,F) vector

    @staticmethod
    def obj_func_consumers_1(x):
        x = np.hstack(x[A10b.F: A10b.F+A10b.C][0]).reshape(A10b.P, -1)
        Q_i = A10b.get_constants(i=1).get('Q_i')
        b_i = A10b.get_constants(i=1).get('b_i')
        return A10b.utility_function(x, Q_i, b_i)

    @staticmethod
    def obj_func_market(x):
        p = x[-1].reshape(-1,1)
        y = np.hstack(x[:A10b.F]).reshape(A10b.P, -1)
        x = np.hstack(x[A10b.F:A10b.F+A10b.C]).reshape(A10b.P, -1)
        xi = A10b.get_xi()
        return p.T @ ( np.sum(x, axis=1, keepdims=True) - np.sum(y, axis=1, keepdims=True) - np.sum(xi, axis=1, keepdims=True) )

    @staticmethod
    def utility_function_1(x_i, a, b, i):
        """
        x_i: np.array shape (P,1)
        a: np.array shape (P,1)
        b: np.array shape (P,1)
        i: integer
        """
        return np.sum((a + i + A10b.F) * np.log(x_i + b + 2 * (i + A10b.F)), axis=0)

    @staticmethod
    def utility_function_1_der(x_i, a, b, i):
        return (a + i + A10b.F) / ((x_i + b + 2 * (i + A10b.F)) * np.log(10) )

    @staticmethod
    def utility_function_2(x_i, c, d, i):
        """
        x_i: np.array shape (P,1)
        c: np.array shape (P,1)
        d: np.array shape (P,1)
        i: integer
        """
        return np.sum((c + i + A10b.F) * np.log(x_i + d + i + A10b.F), axis=0)

    @staticmethod
    def utility_function_2_der(x_i, c, d, i):
        return (c + i + A10b.F) / ((x_i + d + i + A10b.F) * np.log(10))

    @staticmethod
    def obj_func_consumers_der(x):
        x = np.hstack(x[A10b.F: A10b.F + A10b.C][0]).reshape(A10b.P, -1)
        a,b,c,d = A10b.get_constants()
        der_1_10 = []
        der_10_20 = []
        for i in range(int(A10b.C/2)):
            # print(i, i+ int(A10b.C/2) )
            p1_10 = A10b.utility_function_1_der(x, a, b, i)
            p10_20 = A10b.utility_function_2_der(x, c, d, i+ int(A10b.C/2))
            der_1_10.append(p1_10.flatten())
            der_10_20.append(p10_20.flatten())
        obj_der_1 = np.concat(der_1_10).reshape(-1,1)
        obj_der_2 = np.concat(der_10_20).reshape(-1,1)
        obj_der = np.vstack((obj_der_1, obj_der_2))
        return obj_der # shape (P*C, 1)

    @staticmethod
    def obj_func_firms_der(x):
        p = np.tile(x[-1].reshape(-1,1), A10b.F).reshape(-1,1)
        return p # returns (F,1) vector

    @staticmethod
    def obj_func_market_der(x):
        y = np.hstack(x[:A10b.F]).reshape(A10b.P, -1)
        x = np.hstack(x[A10b.F:A10b.F + A10b.C]).reshape(A10b.P, -1)
        xi = A10b.get_xi()
        return np.sum(x, axis=1, keepdims=True) - np.sum(y, axis=1, keepdims=True) - np.sum(xi, axis=1, keepdims=True)

    @staticmethod
    def obj_der(x):
        firms_der = A10b.obj_func_firms_der(x) # (F*P, 1)
        consumers_der = A10b.obj_func_consumers_der(x) # (C*P, 1)
        market_der = A10b.obj_func_market_der(x) # (P, 1)
        return np.vstack((firms_der, consumers_der, market_der)).reshape(-1,1)


    @staticmethod
    def g0(x):
        idx = 10*np.array([i+1 for i in range(A10b.F)]).reshape(-1,1)
        y = np.hstack(x[:A10b.F]).reshape(A10b.P, -1)
        sum_y = np.sum(y,axis=0).reshape(-1,1)**2
        return sum_y - idx

    @staticmethod
    def g1(x):
        p = x[-1].reshape(-1, 1)
        x = np.hstack(x[A10b.F: A10b.F + A10b.C]).reshape(A10b.P, -1)
        xi = A10b.get_xi()
        return (p.T @ x).reshape(-1,1) - (p.T @ xi).reshape(-1,1)

    @staticmethod
    def g2(x):
        p = x[-1].reshape(-1, 1)
        return np.sum(p, axis=0) - 1

    @staticmethod
    def g3(x):
        p = x[-1].reshape(-1, 1)
        return -np.sum(p, axis=0) + 1

    @staticmethod
    def g4(x):
        x = np.concatenate(x).reshape(-1, 1)
        return 0 - x


    @staticmethod
    def g0_der(x):
        y = 2 * x[:A10b.F*A10b.P].reshape(-1, 1)
        pad = np.array([0 for i in range(A10b.C * A10b.P + A10b.P)]).reshape(-1, 1)
        return np.vstack((y, pad))

    @staticmethod
    def g1_der(x):
        zeros = np.zeros_like(x).reshape(-1,1)
        p = x[-A10b.P:].reshape(-1, 1)
        p_stack = np.vstack([p for _ in range(A10b.C)])
        zeros[A10b.F*A10b.P : A10b.F*A10b.P + A10b.P * A10b.C] = p_stack
        return zeros

    @staticmethod
    def g2_der(x):
        return 1

    @staticmethod
    def g3_der(x):
        return -1

    @staticmethod
    def g4_der(x):
        return -1

    @staticmethod
    def get_xi():
        # 1-C/2 is xi1
        # C/2 - C is second xi2
        xi1 = np.array([2, 3, 4, 1, 6]).reshape(-1, 1)
        xi2 = np.array([6, 5, 4, 3, 2]).reshape(-1, 1)
        xi1s = np.hstack([xi1 for _ in range(int(A10b.C/2))]).reshape(A10b.P,-1)
        xi2s = np.hstack([xi2 for _ in range(int(A10b.C/2))]).reshape(A10b.P,-1)
        xi = np.hstack([xi1s, xi2s])
        return xi

    @staticmethod
    def get_constants():
        a = np.array([1,2,4,6,8]).reshape(-1,1)
        b = np.array([20,30,30,40,50]).reshape(-1,1)
        c = np.array([10,6,4,10,1]).reshape(-1,1)
        d = np.array([50,40,30,20,20]).reshape(-1,1)
        return a,b,c,d
#
# vector_sizes = A10b.define_players()[0]
# x = [0 for _ in range(A10b.N * A10b.P)]
# actions = construct_vectors(x, vector_sizes)
# print(A10b.obj_der(actions).shape)