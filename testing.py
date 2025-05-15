from GNESolver5 import *
from Problems.ProblemA8 import A8
from Problems.ProblemA7 import A7
from Problems.ProblemA5 import A5


if __name__ == '__main__':
    problem = A7

    x1 = np.array([[10], [10], [10], [10], [10]], dtype=np.float64)
    x2 = np.array([[10], [10], [10], [10], [10]], dtype=np.float64)
    x3 = np.array([[10], [10], [10], [10], [10]], dtype=np.float64)
    x4 = np.array([[10], [10], [10], [10], [10]], dtype=np.float64)

    print(problem.obj_func_1([x1, x2, x3, x4]))
    print(problem.obj_func_2([x1, x2, x3, x4]))
    print(problem.obj_func_3([x1, x2, x3, x4]))
    print(problem.obj_func_4([x1, x2, x3, x4]))

    x1 = np.array([[10], [10], [10], [10], [10]], dtype=np.float64)
    x2 = np.array([[10], [10], [10], [10], [10]], dtype=np.float64)
    x3 = np.array([[10], [10], [10], [10], [10]], dtype=np.float64)
    x4 = np.array([[10], [10], [10], [10], [10]], dtype=np.float64)
    print(problem.obj_func_der_1([x1, x2, x3, x4]))
    print(problem.obj_func_der_2([x1, x2, x3, x4]))
    print(problem.obj_func_der_3([x1, x2, x3, x4]))
    print(problem.obj_func_der_4([x1, x2, x3, x4]))

    x1 = np.array([[10], [10], [10], [10], [10]], dtype=np.float64)
    x2 = np.array([[10], [10], [10], [10], [10]], dtype=np.float64)
    x3 = np.array([[10], [10], [10], [10], [10]], dtype=np.float64)
    x4 = np.array([[10], [10], [10], [10], [10]], dtype=np.float64)
    x = [x1, x2, x3, x4]
    print(problem.g0(x))
    print(problem.g1(x))
    print(problem.g2(x))
    print(problem.g3(x))


