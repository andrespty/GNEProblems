from GNESolver5 import *
from Problems.ProblemA8 import A8
from Problems.ProblemA7 import A7
from Problems.ProblemA5 import A5
from Problems.ProblemA11 import A11


if __name__ == '__main__':
    problem = A11

    x1 = np.array([[10]], dtype=np.float64)
    x2 = np.array([[10]], dtype=np.float64)

    print(problem.obj_func_1([x1,x2]))
    print(problem.obj_func_2([x1,x2]))
    print(problem.obj_func_der_1([x1,x2]))
    print(problem.obj_func_der_2([x1,x2]))
