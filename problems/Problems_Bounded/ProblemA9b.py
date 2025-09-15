import numpy as np

from tests.Problems_Unbounded.ProblemA16 import player_vector_sizes
from gne_solver.misc import construct_vectors


class A9b:
    K=16
    N=7

    @staticmethod
    def define_players():
        player_vector_sizes = [A9b.K for _ in range(A9b.N)]
        player_objective_functions = [0 for _ in range(A9b.N)]  # change to all 0s
        player_constraints = [[0] for _ in range(A9b.N)]
        bounds = [(0, 100) for _ in range(A9b.N * A9b.K)] + [(0, 100)]
        bounds_training = [(0, 100) for _ in range(A9b.N * A9b.K)] + [(0, 100)]
        return [player_vector_sizes, player_objective_functions, player_constraints, bounds, bounds_training]

    @staticmethod
    def objective_functions():
        return [A9b.obj_func]

    @staticmethod
    def objective_function_derivatives():
        return [A9b.obj_func_der]

    @staticmethod
    def constraints():
        return [A9b.g0]

    @staticmethod
    def constraint_derivatives():
        return [A9b.g0_der]

    @staticmethod
    def obj_func(x):
        # x: numpy array (N, 1)
        # B: constant
        return sum(x)

    @staticmethod
    def obj_func_der(x):
        return np.array([1 for _ in range(A9b.N * A9b.K)]).reshape(-1,1)

    @staticmethod
    def g0_manual(x):
        """
        here for checks only
        """
        X = np.concatenate(x).reshape(-1,1)
        sigma = 0.3162
        values = []
        for vu in range(A9b.N):
            L = 8
            H = A9b.get_h_v(vu).reshape(A9b.N, A9b.K)
            hx = H[vu].reshape(-1, 1) * x[vu]

            H_ni = np.delete(H, vu, axis=0).reshape(-1,1)
            X_ni = np.delete(X, slice(vu * A9b.K, (vu + 1) * A9b.K), axis=0)

            hx_ni = np.sum((H_ni * X_ni).reshape(A9b.N-1, A9b.K), axis=0).reshape(-1,1)
            # print(hx_ni)
            constraint = np.log2( 1 + (hx/(sigma**2 + hx_ni)) ) - L
            values.append(np.sum(constraint).flatten())
        return np.concatenate(values).reshape(-1,1)

    @staticmethod
    def g0(x):
        K, N = A9b.K, A9b.N
        sigma = 0.3162
        L = 16

        # Flatten input

        X = np.concatenate(x).reshape(N * K, 1)  # (K*N, 1)
        H_all = np.stack([A9b.get_h_v(vu).reshape(N, K) for vu in range(N)], axis=0)  # shape: (N, N, K)

        # Reshape X to (N, K, 1) for broadcasting
        X_split = np.concatenate(x).reshape(N, K, 1)  # shape: (N, K, 1)

        # Compute elementwise product: shape (N, N, K, 1)
        HX_all = H_all[..., :, np.newaxis] * X_split[np.newaxis, ...]

        # Split signal vs interference:
        # Signal for vu is at H_all[vu, vu] * x[vu] → shape (N, K, 1)
        signal = np.array([HX_all[i, i] for i in range(N)])  # shape (N, K, 1)

        # Sum all players' contributions (axis=1), then subtract self
        total_hx = np.sum(HX_all, axis=1)  # shape: (N, K, 1)
        interference = total_hx - signal  # shape: (N, K, 1)

        # Compute constraint
        constraint = np.log2(1 + (signal / (sigma ** 2 + interference))) - L  # shape: (N, K, 1)

        # Sum over K dimensions per player
        return np.sum(constraint, axis=1).reshape(-1,1)  # shape: (N,)

    @staticmethod
    def g0_der(x):
        sigma = 0.3162
        result = []
        for vu in range(A9b.N):
            H_v = A9b.get_h_v(vu).reshape(A9b.N, A9b.K)
            players = x.reshape(A9b.N, A9b.K)
            H_vv = H_v[vu].reshape(-1, 1)
            D = (sigma ** 2) + np.sum(players * H_v, axis=0).reshape(-1,1)
            grad = (1/np.log(2))*(H_vv / D)
            result.append(grad.ravel())
        result = np.concatenate(result).reshape(-1, 1)
        return result

    @staticmethod
    def get_h_v(vu):
        if vu > A9b.N - 1:
            print('Cant be done')
            return 1
        H = A9b.h_matrix()
        # padding = mu * A9b.K
        return H[:, vu].reshape(-1,1)

    @staticmethod
    def h_matrix():
        return np.array([
    [0.0129, 0.0010, 0.0015, 0.0008, 0.0005, 0.0088, 0.0048],
    [0.0037, 0.0062, 0.0020, 0.0044, 0.0043, 0.0040, 0.0029],
    [0.0514, 0.0114, 0.0024, 0.0094, 0.0063, 0.0057, 0.0045],
    [0.1382, 0.0087, 0.0096, 0.0113, 0.0040, 0.0144, 0.0114],
    [0.1824, 0.0026, 0.0180, 0.0068, 0.0054, 0.0140, 0.0157],
    [0.1193, 0.0002, 0.0135, 0.0007, 0.0094, 0.0043, 0.0132],
    [0.0290, 0.0002, 0.0033, 0.0019, 0.0116, 0.0016, 0.0089],
    [0.0188, 0.0020, 0.0016, 0.0075, 0.0192, 0.0065, 0.0105],
    [0.0550, 0.0074, 0.0038, 0.0070, 0.0270, 0.0076, 0.0157],
    [0.0642, 0.0134, 0.0024, 0.0020, 0.0194, 0.0056, 0.0136],
    [0.0495, 0.0166, 0.0004, 0.0019, 0.0052, 0.0058, 0.0041],
    [0.0318, 0.0160, 0.0009, 0.0042, 0.0017, 0.0067, 0.0001],
    [0.0287, 0.0098, 0.0050, 0.0027, 0.0030, 0.0050, 0.0027],
    [0.0585, 0.0025, 0.0102, 0.0004, 0.0010, 0.0020, 0.0021],
    [0.0843, 0.0001, 0.0092, 0.0002, 0.0002, 0.0027, 0.0000],
    [0.0586, 0.0002, 0.0032, 0.0001, 0.0002, 0.0077, 0.0027],
    [0.0028, 0.0288, 0.0208, 0.0004, 0.0000, 0.0002, 0.0011],
    [0.0088, 0.0500, 0.0094, 0.0001, 0.0002, 0.0003, 0.0017],
    [0.0039, 0.0251, 0.0125, 0.0000, 0.0002, 0.0006, 0.0010],
    [0.0016, 0.0091, 0.0147, 0.0000, 0.0001, 0.0003, 0.0003],
    [0.0061, 0.1168, 0.0130, 0.0005, 0.0000, 0.0001, 0.0005],
    [0.0098, 0.1155, 0.0088, 0.0009, 0.0001, 0.0002, 0.0002],
    [0.0121, 0.0029, 0.0044, 0.0006, 0.0001, 0.0001, 0.0008],
    [0.0142, 0.1913, 0.0019, 0.0001, 0.0003, 0.0000, 0.0021],
    [0.0106, 0.4811, 0.0009, 0.0001, 0.0004, 0.0001, 0.0011],
    [0.0035, 0.3142, 0.0017, 0.0002, 0.0004, 0.0003, 0.0002],
    [0.0004, 0.0131, 0.0044, 0.0000, 0.0003, 0.0006, 0.0016],
    [0.0002, 0.1289, 0.0049, 0.0002, 0.0002, 0.0007, 0.0014],
    [0.0009, 0.3551, 0.0038, 0.0009, 0.0001, 0.0004, 0.0002],
    [0.0031, 0.2986, 0.0099, 0.0014, 0.0001, 0.0001, 0.0017],
    [0.0034, 0.1287, 0.0202, 0.0010, 0.0000, 0.0003, 0.0029],
    [0.0023, 0.0011, 0.0609, 0.0017, 0.0002, 0.0001, 0.0002],
    [0.0028, 0.0021, 0.0588, 0.0012, 0.0001, 0.0004, 0.0002],
    [0.0025, 0.0071, 0.0554, 0.0028, 0.0001, 0.0004, 0.0001],
    [0.0008, 0.0096, 0.1574, 0.0099, 0.0000, 0.0002, 0.0000],
    [0.0007, 0.0045, 0.2024, 0.0192, 0.0002, 0.0001, 0.0000],
    [0.0036, 0.0003, 0.0726, 0.0165, 0.0005, 0.0000, 0.0001],
    [0.0051, 0.0025, 0.0497, 0.0032, 0.0004, 0.0000, 0.0004],
    [0.0024, 0.0038, 0.1986, 0.0029, 0.0004, 0.0001, 0.0004],
    [0.0000, 0.0016, 0.2281, 0.0205, 0.0005, 0.0002, 0.0002],
    [0.0008, 0.0005, 0.1334, 0.0312, 0.0004, 0.0004, 0.0000],
    [0.0012, 0.0007, 0.0762, 0.0231, 0.0003, 0.0006, 0.0001],
    [0.0006, 0.0005, 0.0209, 0.0103, 0.0004, 0.0004, 0.0001],
    [0.0021, 0.0005, 0.0115, 0.0042, 0.0002, 0.0001, 0.0001],
    [0.0043, 0.0004, 0.1096, 0.0032, 0.0000, 0.0003, 0.0001],
    [0.0041, 0.0005, 0.1429, 0.0034, 0.0002, 0.0004, 0.0001],
    [0.0026, 0.0012, 0.0759, 0.0028, 0.0004, 0.0001, 0.0002],
    [0.0006, 0.0001, 0.0016, 0.0492, 0.0066, 0.0011, 0.0002],
    [0.0002, 0.0002, 0.0014, 0.0128, 0.0003, 0.0008, 0.0002],
    [0.0007, 0.0002, 0.0013, 0.1154, 0.0011, 0.0000, 0.0001],
    [0.0005, 0.0001, 0.0018, 0.1459, 0.0010, 0.0003, 0.0001],
    [0.0011, 0.0000, 0.0010, 0.0509, 0.0017, 0.0005, 0.0005],
    [0.0016, 0.0000, 0.0001, 0.0136, 0.0077, 0.0001, 0.0006],
    [0.0017, 0.0000, 0.0000, 0.0487, 0.0095, 0.0001, 0.0002],
    [0.0014, 0.0000, 0.0003, 0.0591, 0.0041, 0.0004, 0.0000],
    [0.0003, 0.0002, 0.0017, 0.0434, 0.0016, 0.0002, 0.0003],
    [0.0003, 0.0004, 0.0030, 0.0570, 0.0015, 0.0001, 0.0005],
    [0.0027, 0.0004, 0.0016, 0.1731, 0.0001, 0.0004, 0.0003],
    [0.0030, 0.0001, 0.0005, 0.3232, 0.0019, 0.0003, 0.0000],
    [0.0007, 0.0000, 0.0025, 0.2794, 0.0055, 0.0001, 0.0001],
    [0.0019, 0.0001, 0.0030, 0.0914, 0.0076, 0.0002, 0.0002],
    [0.0051, 0.0002, 0.0008, 0.0519, 0.0119, 0.0000, 0.0001],
    [0.0039, 0.0001, 0.0004, 0.1091, 0.0139, 0.0004, 0.0000],
    [0.0000, 0.0001, 0.0000, 0.0003, 0.0301, 0.0023, 0.0003],
    [0.0003, 0.0001, 0.0000, 0.0010, 0.0167, 0.0070, 0.0002],
    [0.0008, 0.0000, 0.0001, 0.0018, 0.0398, 0.0089, 0.0009],
    [0.0007, 0.0001, 0.0003, 0.0021, 0.0606, 0.0049, 0.0021],
    [0.0004, 0.0003, 0.0005, 0.0014, 0.0857, 0.0010, 0.0015],
    [0.0002, 0.0001, 0.0002, 0.0003, 0.1207, 0.0012, 0.0003],
    [0.0002, 0.0001, 0.0001, 0.0002, 0.0936, 0.0021, 0.0005],
    [0.0005, 0.0008, 0.0003, 0.0007, 0.0217, 0.0026, 0.0007],
    [0.0004, 0.0011, 0.0005, 0.0008, 0.0188, 0.0035, 0.0002],
    [0.0001, 0.0005, 0.0003, 0.0003, 0.0808, 0.0027, 0.0003],
    [0.0004, 0.0000, 0.0001, 0.0002, 0.1234, 0.0007, 0.0003],
    [0.0007, 0.0001, 0.0000, 0.0006, 0.1248, 0.0013, 0.0000],
    [0.0004, 0.0000, 0.0002, 0.0007, 0.0850, 0.0029, 0.0011],
    [0.0002, 0.0000, 0.0003, 0.0003, 0.0287, 0.0024, 0.0022]])

x1 = np.array([1,1,1,1,1,1,1,1]).reshape(-1,1)
x2 = np.array([2,2,2,2,2,2,2,2]).reshape(-1,1)
x3 = np.array([3,3,3,3,3,3,3,3]).reshape(-1,1)
x4 = np.array([4,4,4,4,4,4,4,4]).reshape(-1,1)
x5 = np.array([5,5,5,5,5,5,5,5]).reshape(-1,1)
x6 = np.array([6,6,6,6,6,6,6,6]).reshape(-1,1)
x7 = np.array([7,7,7,7,7,7,7,7]).reshape(-1,1)

x = np.vstack([x1,x2,x3,x4,x5,x6,x7]).reshape(-1,1)
player_vector_sizes = [A9b.K for _ in range(A9b.N)]
print(A9b.g0_manual(construct_vectors(x,player_vector_sizes)))
print(A9b.obj_func_der(x).shape)
