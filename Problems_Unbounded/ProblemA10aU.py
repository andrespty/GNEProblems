import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt

class A10aU:
    F= 2
    C= 5
    P= 3
    N = F + C + 1

    @staticmethod
    def define_players():
        player_vector_sizes = [A10aU.P for _ in range(A10aU.N)]
        player_objective_functions = [0 for _ in range(A10aU.N)]  # change to all 0s
        player_constraints = [[0,4], [0,4], [1,4],[1,4],[1,4],[1,4],[1,4], [2,3,4]]
        return [player_vector_sizes, player_objective_functions, player_constraints]

    @staticmethod
    def objective_functions():
        return [A10aU.obj_func]

    @staticmethod
    def objective_function_derivatives():
        return [A10aU.obj_func_der]

    @staticmethod
    def constraints():
        return [A10aU.g0, A10aU.g1, A10aU.g2, A10aU.g3, A10aU.g4]

    @staticmethod
    def constraint_derivatives():
        return [A10aU.g0_der, A10aU.g1_der, A10aU.g2_der, A10aU.g3_der, A10aU.g4_der]

    # SET UP FINISHED

    @staticmethod
    def utility_function(x_i, Q_i, b_i):
        return -0.5 * x_i.T @ Q_i @ x_i + b_i.T @ x_i

    @staticmethod
    def obj_func_firms(x):
        p = x[-1].reshape(-1,1)
        y = np.hstack(x[:A10aU.F]).reshape(A10aU.P, -1)
        return (p.T @ y).reshape(-1,1) # returns (1,F) vector

    @staticmethod
    def obj_func_consumers(x):
        x = np.hstack(x[A10aU.F: A10aU.F + A10aU.C]).reshape(A10aU.P, -1)
        obj = []
        for i in range(A10aU.C):
            Q_i = A10aU.get_constants(i=i+1).get('Q_i')
            b_i = A10aU.get_constants(i=i+1).get('b_i')
            u = -A10aU.utility_function(x[:,i].reshape(-1,1), Q_i, b_i)
            obj.append(u)
        return np.array(obj).reshape(-1,1)

    @staticmethod
    def obj_func_market(x):
        p = x[-1].reshape(-1,1)     # take last player actions and shape them as a vector
        y = np.hstack(x[:A10aU.F]).reshape(A10aU.P, -1)
        x = np.hstack(x[A10aU.F:A10aU.F+A10aU.C]).reshape(A10aU.P, -1)
        xi = A10aU.get_xi()
        return (p.T @ ( np.sum(x, axis=1, keepdims=True) - np.sum(y, axis=1, keepdims=True) - np.sum(xi, axis=1, keepdims=True) )).reshape(-1,1)

    @staticmethod
    def obj_func(x):
        firms = A10aU.obj_func_firms(x)
        consumers = A10aU.obj_func_consumers(x)
        market = A10aU.obj_func_market(x)
        return np.vstack((firms, consumers, market)).reshape(-1,1)

    @staticmethod
    def utility_function_der(x_i, Q_i, b_i):
        return - Q_i @ x_i + b_i

    @staticmethod
    def obj_func_consumers_der(x):
        x = np.hstack(x[A10aU.F: A10aU.F + A10aU.C]).reshape(A10aU.P, -1)
        der = []
        for i in range(A10aU.C):
            Q_i = A10aU.get_constants(i=i + 1).get('Q_i')
            b_i = A10aU.get_constants(i=i + 1).get('b_i')
            u_der = A10aU.utility_function_der(x[:, i].reshape(-1, 1), Q_i, b_i)
            der.append(u_der)
        return np.concatenate(der).reshape(-1,1)

    @staticmethod
    def obj_func_firms_der(x):
        p = np.tile(x[-1].reshape(-1, 1), A10aU.F).reshape(-1, 1)
        return p  # returns (F,1) vector

    @staticmethod
    def obj_func_market_der(x):
        y = np.hstack(x[:A10aU.F]).reshape(A10aU.P, -1)
        x = np.hstack(x[A10aU.F:A10aU.F + A10aU.C]).reshape(A10aU.P, -1)
        xi = A10aU.get_xi()
        return np.sum(x, axis=1, keepdims=True) - np.sum(y, axis=1, keepdims=True) - np.sum(xi, axis=1, keepdims=True)

    @staticmethod
    def obj_func_der(x):
        firms_der = A10aU.obj_func_firms_der(x)
        consumers_der = A10aU.obj_func_consumers_der(x)
        market_der = A10aU.obj_func_market_der(x)
        return np.vstack((firms_der, consumers_der, market_der)).reshape(-1,1)

    @staticmethod
    def g0(x):
        idx = 10*np.array([i+1 for i in range(A10aU.F)]).reshape(-1,1)
        y = np.hstack(x[:A10aU.F]).reshape(A10aU.P, -1)**2
        sum_y = np.sum(y,axis=0).reshape(-1,1)
        return sum_y - idx

    @staticmethod
    def g1(x):
        p = x[-1].reshape(-1, 1)
        x = np.hstack(x[A10aU.F: A10aU.F + A10aU.C]).reshape(A10aU.P, -1)
        xi = A10aU.get_xi()
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
        X = np.concatenate(x).reshape(-1,1)
        return 0 - X

    @staticmethod
    def g0_der(x):
        y = 2 * x[:A10aU.F*A10aU.P].reshape(-1, 1)
        pad = np.array([0 for i in range(A10aU.C * A10aU.P + A10aU.P)]).reshape(-1, 1)
        return np.vstack((y, pad))

    @staticmethod
    def g1_der(x):
        zeros = np.zeros_like(x).reshape(-1,1)
        p = x[-A10aU.P:].reshape(-1, 1)
        p_stack = np.vstack([p for _ in range(A10aU.C)])
        zeros[A10aU.F*A10aU.P : A10aU.F*A10aU.P + A10aU.P * A10aU.C] = p_stack
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
        x1 = np.array([2, 3, 4]).reshape(-1, 1)
        x2 = np.array([2, 3, 4]).reshape(-1, 1)
        x3 = np.array([6, 5, 4]).reshape(-1, 1)
        x4 = np.array([6, 5, 4]).reshape(-1, 1)
        x5 = np.array([6, 5, 4]).reshape(-1, 1)
        xi = np.hstack([x1,x2,x3,x4,x5]).reshape(A10aU.P, -1 )
        return xi

    @staticmethod
    def get_constants(i=1):
        if i in [1,2]:
            Q_i = np.array([
                [6, -2, 5],
                [-2, 6, -7],
                [5, -7, 20]
            ])
            b_i = np.array([30+i+A10aU.F,30+i+A10aU.F,30+i+A10aU.F]).reshape(-1,1)
        elif i in [3,4,5]:
            Q_i = np.array([
                [6, 1, 0],
                [1, 7, -5],
                [0, -5, 7]
            ])
            b_i = np.array([30 + 2*(i + A10aU.F), 30 + 2*(i + A10aU.F), 30 + 2*(i + A10aU.F)]).reshape(-1, 1)
        return dict(Q_i=Q_i, b_i=b_i)


