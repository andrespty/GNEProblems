import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt
from library.misc import construct_vectors


class A10c:
    F= 6
    C= 30
    P= 6
    N = F + C + 1

    @staticmethod
    def define_players():
        player_vector_sizes = [A10c.P for _ in range(A10c.N)]
        player_objective_functions = [0 for _ in range(A10c.N)]  # change to all 0s
        f_player_constraints = [[0] for _ in range(A10c.F)]
        c_player_constraints = [[1] for _ in range(A10c.C)]
        player_constraints = f_player_constraints + c_player_constraints + [[2,3]]
        return [player_vector_sizes, player_objective_functions, player_constraints]

    @staticmethod
    def objective_functions():
        return [A10c.obj_func_firms,
                A10c.obj_func_market
        ]

    @staticmethod
    def objective_function_derivatives():
        return [A10c.obj_der]

    @staticmethod
    def constraints():
        return [A10c.g0, A10c.g1, A10c.g2, A10c.g3, A10c.g4]

    @staticmethod
    def constraint_derivatives():
        return [A10c.g0_der, A10c.g1_der, A10c.g2_der, A10c.g3_der, A10c.g4_der]

    @staticmethod
    def obj_func_firms(x):
        p = x[-1].reshape(-1,1)
        y = np.hstack(x[:A10c.F]).reshape(A10c.P, -1)
        return p.T @ y # returns (1,F) vector

    @staticmethod
    def obj_func_consumers(x):
        x = np.hstack(x[A10c.F: A10c.F+A10c.C][0]).reshape(A10c.P, -1)
        Q_i = A10c.get_constants(i=1).get('Q_i')
        b_i = A10c.get_constants(i=1).get('b_i')
        return A10c.utility_function(x, Q_i, b_i)

    @staticmethod
    def obj_func_market(x):
        p = x[-1].reshape(-1,1)
        y = np.hstack(x[:A10c.F]).reshape(A10c.P, -1)
        x = np.hstack(x[A10c.F:A10c.F+A10c.C]).reshape(A10c.P, -1)
        xi = A10c.get_xi()
        return p.T @ ( np.sum(x, axis=1, keepdims=True) - np.sum(y, axis=1, keepdims=True) - np.sum(xi, axis=1, keepdims=True) )

    @staticmethod
    def utility_function(x_i, Q_i, b_i):
        return -0.5 * x_i.T @ Q_i @ x_i + b_i.T @ x_i

    @staticmethod
    def utility_function_der(x_i, Q_i, b_i):
        return - Q_i @ x_i + b_i

    @staticmethod
    def obj_func_consumers_der(x):
        x = np.hstack(x[A10c.F: A10c.F + A10c.C]).reshape(A10c.P, -1)
        der_1_15 = []
        der_15_30 = []
        for i in range(int(A10c.C/2)):
            A, b_i = A10c.get_constants(i+1)
            B, bb_i = A10c.get_constants(i+1 + int(A10c.C/2))
            p1_15 = A10c.utility_function_der(x[:,i].reshape(-1,1), A, b_i)
            p15_30 = A10c.utility_function_der(x[:, i+ int(A10c.C/2)].reshape(-1,1), B, bb_i)
            der_1_15.append(p1_15.flatten())
            der_15_30.append(p15_30.flatten())
        obj_der_1 = np.concat(der_1_15).reshape(-1,1)
        obj_der_2 = np.concat(der_15_30).reshape(-1,1)
        obj_der = np.vstack((obj_der_1, obj_der_2))
        return obj_der # shape (P*C, 1)

    @staticmethod
    def obj_func_firms_der(x):
        p = np.tile(x[-1].reshape(-1,1), A10c.F).reshape(-1,1)
        return p # returns (F,1) vector

    @staticmethod
    def obj_func_market_der(x):
        y = np.hstack(x[:A10c.F]).reshape(A10c.P, -1)
        x = np.hstack(x[A10c.F:A10c.F + A10c.C]).reshape(A10c.P, -1)
        xi = A10c.get_xi()
        return np.sum(x, axis=1, keepdims=True) - np.sum(y, axis=1, keepdims=True) - np.sum(xi, axis=1, keepdims=True)

    @staticmethod
    def obj_der(x):
        firms_der = A10c.obj_func_firms_der(x) # (F*P, 1)
        consumers_der = A10c.obj_func_consumers_der(x) # (C*P, 1)
        market_der = A10c.obj_func_market_der(x) # (P, 1)
        return np.vstack((firms_der, consumers_der, market_der)).reshape(-1,1)


    @staticmethod
    def g0(x):
        idx = 10*np.array([i+1 for i in range(A10c.F)]).reshape(-1,1)
        y = np.hstack(x[:A10c.F]).reshape(A10c.P, -1)
        sum_y = np.sum(y,axis=0).reshape(-1,1)**2
        return sum_y - idx

    @staticmethod
    def g1(x):
        p = x[-1].reshape(-1, 1)
        x = np.hstack(x[A10c.F: A10c.F + A10c.C]).reshape(A10c.P, -1)
        xi = A10c.get_xi()
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
        y = 2 * x[:A10c.F*A10c.P].reshape(-1, 1)
        pad = np.array([0 for i in range(A10c.C * A10c.P + A10c.P)]).reshape(-1, 1)
        return np.vstack((y, pad))

    @staticmethod
    def g1_der(x):
        zeros = np.zeros_like(x).reshape(-1,1)
        p = x[-A10c.P:].reshape(-1, 1)
        p_stack = np.vstack([p for _ in range(A10c.C)])
        zeros[A10c.F*A10c.P : A10c.F*A10c.P + A10c.P * A10c.C] = p_stack
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
        xi1 = np.array([2, 3, 4, 1, 6, 1]).reshape(-1, 1)
        xi2 = np.array([6, 5, 4, 3, 2, 8]).reshape(-1, 1)
        xi1s = np.hstack([xi1 for _ in range(int(A10c.C/2))]).reshape(A10c.P,-1)
        xi2s = np.hstack([xi2 for _ in range(int(A10c.C/2))]).reshape(A10c.P,-1)
        xi = np.hstack([xi1s, xi2s])
        return xi

    @staticmethod
    def get_constants(i=1):
        if i <= A10c.C:
            A = [
                [68.22249416536778, 12.12481199690621, -8.35496210217478, -6.81177486915109, -4.66752803051747, 3.64100170417482],
                [12.12481199690621, 53.51450780426463, -21.77618227261339, -15.00376305863444, -0.11788350473544, 2.03354709400720],
                [-8.35496210217478, -21.77618227261339, 35.44033408387684, 4.35160649036518, 19.17472558234163, -3.40090742729160],
                [-6.81177486915109, -15.00376305863444, 4.35160649036518, 52.25155022199242, -5.99490328518247, 20.40443259092577],
                [-4.66752803051747, -0.11788350473544, 19.17472558234163, -5.99490328518247, 23.32798561358070, -3.58535668529727],
                [3.64100170417482, 2.03354709400720, -3.40090742729160, 20.40443259092577, -3.58535668529727, 10.21258119890765]
            ]
            b_i = np.array([50 + i + A10c.F, 60 + i + A10c.F, 70 + i + A10c.F, 60 + i + A10c.F, 60 + i + A10c.F, 50 + i + A10c.F]).reshape(-1, 1)
            return [np.array(A), b_i]
        else:
            B  = [
                [61.74633559943146, -23.83006225091380, 16.78581949473039, 14.42073900860500, -2.75188745616575, 13.44307656650567],
                [-23.83006225091380, 37.64246654306209, -3.76510322128227, 16.32022449045404, -39.90743633716275, 11.38657250296817],
                [16.78581949473039, -3.76510322128227, 53.34843665848310, 4.60388415537161, -23.04611587657949, -25.31392346426841],
                [14.42073900860500, 16.32022449045404, 4.60388415537161, 40.69699687713468, -30.78019133996427, 17.08866411420883],
                [-2.75188745616575, -39.90743633716275, -23.04611587657949, -30.78019133996427, 66.22678445157413, -12.28091080313848],
                [13.44307656650567, 11.38657250296817, -25.31392346426841, 17.08866411420883, -12.28091080313848, 41.37849544246254]
            ]
            b_i = np.array([50 + 2 * (i + A10c.F), 60 + 2 * (i + A10c.F), 50 + 2 * (i + A10c.F), 70 + 2 * (i + A10c.F), 70 + 2 * (i + A10c.F), 60 + 2 * (i + A10c.F)]).reshape(-1, 1)

            return [np.array(B), b_i]
#
vector_sizes = A10c.define_players()[0]
x = [0 for _ in range(A10c.N * A10c.P)]
actions = construct_vectors(x, vector_sizes)
print(A10c.obj_der(actions).shape)