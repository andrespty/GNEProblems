from GNESolver5 import *
from Problems.ProblemA1 import A1
from Problems.ProblemA8 import A8
from Problems.ProblemA7 import A7
from Problems.ProblemA5 import A5
from Problems.ProblemA11 import A11


if __name__ == '__main__':
    problem = A5

    x1 = np.array([[10,10,10]], dtype=np.float64).reshape(-1,1)
    x2 = np.array([[10,10]], dtype=np.float64).reshape(-1,1)
    x3 = np.array([[10,10]], dtype=np.float64).reshape(-1,1)

    print(problem.obj_func_1([x1,x2,x3]))
    # print(problem.obj_func_2([x1,x2]))
    # print(problem.obj_func_der_1([x1,x2]))
    # print(problem.obj_func_der_2([x1,x2]))

    x1 = np.array([[10]], dtype=np.float64).reshape(-1, 1)
    x2 = np.array([[10]], dtype=np.float64).reshape(-1, 1)
    x3 = np.array([[10]], dtype=np.float64).reshape(-1, 1)
    x4 = np.array([[10]], dtype=np.float64).reshape(-1, 1)
    x5 = np.array([[10]], dtype=np.float64).reshape(-1, 1)
    x6 = np.array([[10]], dtype=np.float64).reshape(-1, 1)
    x7 = np.array([[10]], dtype=np.float64).reshape(-1, 1)
    x8 = np.array([[10]], dtype=np.float64).reshape(-1, 1)
    x9 = np.array([[10]], dtype=np.float64).reshape(-1, 1)
    x10 = np.array([[10]], dtype=np.float64).reshape(-1, 1)
    problem = A1
    print('Problem',problem.obj_func_der([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]))
