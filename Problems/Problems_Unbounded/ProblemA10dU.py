import numpy as np


class A10dU:
    F= 6
    C= 30
    P= 10
    N = F + C + 1

    @staticmethod
    def define_players():
        player_vector_sizes = [A10dU.P for _ in range(A10dU.N)]
        player_objective_functions = [0 for _ in range(A10dU.N)]  # change to all 0s

        f_player_constraints = [[0] for _ in range(A10dU.F)]
        c_player_constraints = [[1] for _ in range(A10dU.C)]
        player_constraints = f_player_constraints + c_player_constraints + [[2,3]]
        return [player_vector_sizes, player_objective_functions, player_constraints]

    @staticmethod
    def objective_functions():
        return [A10dU.obj_func]

    @staticmethod
    def objective_function_derivatives():
        return [A10dU.obj_der]

    @staticmethod
    def constraints():
        return [A10dU.g0, A10dU.g1, A10dU.g2, A10dU.g3, A10dU.g4]

    @staticmethod
    def constraint_derivatives():
        return [A10dU.g0_der, A10dU.g1_der, A10dU.g2_der, A10dU.g3_der, A10dU.g4_der]

    # SET UP FINISHED

    @staticmethod
    def utility_function_1(x_i, a, b, i):
        """
        x_i: np.array shape (P,1)
        a: np.array shape (P,1)
        b: np.array shape (P,1)
        i: integer
        """
        return np.sum((a + i + A10dU.F) * np.log(x_i + b + 2 * (i + A10dU.F)), axis=0)

    @staticmethod
    def utility_function_2(x_i, c, d, i):
        """
        x_i: np.array shape (P,1)
        c: np.array shape (P,1)
        d: np.array shape (P,1)
        i: integer
        """
        return np.sum((c + i + A10dU.F) * np.log(x_i + d + i + A10dU.F), axis=0)

    @staticmethod
    def obj_func_firms(x):
        p = x[-1].reshape(-1,1)
        y = np.hstack(x[:A10dU.F]).reshape(A10dU.P, -1)
        return (p.T @ y).reshape(-1,1) # returns (1,F) vector

    @staticmethod
    def obj_func_consumers(x):
        x = np.hstack(x[A10dU.F: A10dU.F+A10dU.C]).reshape(A10dU.P, -1)
        a, b, c, d = A10dU.get_constants()
        obj1 = []
        obj2 = []
        for i in range(int(A10dU.C / 2)):
            u1 = A10dU.utility_function_1(x[:, i].reshape(-1, 1), a, b, i + 1)
            u2 = A10dU.utility_function_2(x[:, i + int(A10dU.C / 2)].reshape(-1, 1), c, d, i + 1 + int(A10dU.C / 2))
            obj1.append(u1)
            obj2.append(u2)
        obj1 = np.array(obj1).reshape(-1, 1)
        obj2 = np.array(obj2).reshape(-1, 1)
        return np.vstack((obj1, obj2)).reshape(-1, 1)

    @staticmethod
    def obj_func_market(x):
        p = x[-1].reshape(-1,1)
        y = np.hstack(x[:A10dU.F]).reshape(A10dU.P, -1)
        x = np.hstack(x[A10dU.F:A10dU.F+A10dU.C]).reshape(A10dU.P, -1)
        xi = A10dU.get_xi()
        return p.T @ ( np.sum(x, axis=1, keepdims=True) - np.sum(y, axis=1, keepdims=True) - np.sum(xi, axis=1, keepdims=True) )

    @staticmethod
    def obj_func(x):
        firms = A10dU.obj_func_firms(x)
        consumers = A10dU.obj_func_consumers(x)
        market = A10dU.obj_func_market(x)
        return np.vstack((firms, consumers, market)).reshape(-1, 1)

    # DERIVATIVES

    @staticmethod
    def utility_function_1_der(x_i, a, b, i):
        return (a + i + A10dU.F) / ((x_i + b + 2 * (i + A10dU.F)) * np.log(10) )

    @staticmethod
    def utility_function_2_der(x_i, c, d, i):
        return (c + i + A10dU.F) / ((x_i + d + i + A10dU.F) * np.log(10))

    @staticmethod
    def obj_func_consumers_der(x):
        x = np.hstack(x[A10dU.F: A10dU.F + A10dU.C]).reshape(A10dU.P, -1)
        a,b,c,d = A10dU.get_constants()
        der_1_10 = []
        der_10_20 = []
        for i in range(int(A10dU.C/2)):
            # print(i, i+ int(A10dU.C/2) )
            p1_10 = A10dU.utility_function_1_der(x[:,i].reshape(-1,1), a, b, i+1)
            p10_20 = A10dU.utility_function_2_der(x[:,i+int(A10dU.C/2)].reshape(-1,1), c, d, i+1+ int(A10dU.C/2))
            der_1_10.append(p1_10.flatten())
            der_10_20.append(p10_20.flatten())
        obj_der_1 = np.concat(der_1_10).reshape(-1,1)
        obj_der_2 = np.concat(der_10_20).reshape(-1,1)
        obj_der = np.vstack((obj_der_1, obj_der_2))
        return obj_der # shape (P*C, 1)

    @staticmethod
    def obj_func_firms_der(x):
        p = np.tile(x[-1].reshape(-1,1), A10dU.F).reshape(-1,1)
        return p # returns (F,1) vector

    @staticmethod
    def obj_func_market_der(x):
        y = np.hstack(x[:A10dU.F]).reshape(A10dU.P, -1)
        x = np.hstack(x[A10dU.F:A10dU.F + A10dU.C]).reshape(A10dU.P, -1)
        xi = A10dU.get_xi()
        return np.sum(x, axis=1, keepdims=True) - np.sum(y, axis=1, keepdims=True) - np.sum(xi, axis=1, keepdims=True)

    @staticmethod
    def obj_der(x):
        firms_der = A10dU.obj_func_firms_der(x) # (F*P, 1)
        consumers_der = A10dU.obj_func_consumers_der(x) # (C*P, 1)
        market_der = A10dU.obj_func_market_der(x) # (P, 1)
        return np.vstack((firms_der, consumers_der, market_der)).reshape(-1,1)


    @staticmethod
    def g0(x):
        idx = 10*np.array([i+1 for i in range(A10dU.F)]).reshape(-1,1)
        y = np.hstack(x[:A10dU.F]).reshape(A10dU.P, -1)**2
        sum_y = np.sum(y,axis=0).reshape(-1,1)
        return sum_y - idx

    @staticmethod
    def g1(x):
        p = x[-1].reshape(-1, 1)
        x = np.hstack(x[A10dU.F: A10dU.F + A10dU.C]).reshape(A10dU.P, -1)
        xi = A10dU.get_xi()
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
        y = 2 * x[:A10dU.F*A10dU.P].reshape(-1, 1)
        pad = np.array([0 for i in range(A10dU.C * A10dU.P + A10dU.P)]).reshape(-1, 1)
        return np.vstack((y, pad))

    @staticmethod
    def g1_der(x):
        zeros = np.zeros_like(x).reshape(-1,1)
        p = x[-A10dU.P:].reshape(-1, 1)
        p_stack = np.vstack([p for _ in range(A10dU.C)])
        zeros[A10dU.F*A10dU.P : A10dU.F*A10dU.P + A10dU.P * A10dU.C] = p_stack
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
        xi1 = np.array([2, 3, 4, 1, 6, 1, 3, 6, 2, 10]).reshape(-1, 1)
        xi2 = np.array([6, 5, 4, 3, 2, 8, 4, 6, 2, 0]).reshape(-1, 1)
        xi1s = np.hstack([xi1 for _ in range(int(A10dU.C/2))]).reshape(A10dU.P,-1)
        xi2s = np.hstack([xi2 for _ in range(int(A10dU.C/2))]).reshape(A10dU.P,-1)
        xi = np.hstack([xi1s, xi2s])
        return xi

    @staticmethod
    def get_constants():
        a = np.array([1,2,4,6,8,7,8,10,1,5]).reshape(-1,1)
        b = np.array([50,60,70,60,60,50,50,80,60,70]).reshape(-1,1)
        c = np.array([10,6,4,10,1,2,6,4,9,4]).reshape(-1,1)
        d = np.array([50,60,50,70,70,60,50,50,80,50]).reshape(-1,1)
        return a,b,c,d
#
# vector_sizes = A10dU.define_players()[0]
# x = [0 for _ in range(A10dU.N * A10dU.P)]
# actions = construct_vectors(x, vector_sizes)
# print(A10dU.obj_der(actions).shape)