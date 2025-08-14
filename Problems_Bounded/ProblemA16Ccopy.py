import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt


class A16C:
    @staticmethod
    def papersolution():
        value_1 = [10.40385838022815, 13.03589735757035, 15.40740737819953, 17.38156802485601, 4218.77134655563816 ]
        value_2 = [14.05009569254498, 17.79839784112484, 20.90720382090469, 23.11144740620916, 24.13291790632350 ]
        value_3 = [23.58870405660235, 28.68433662443821, 32.02151725264314, 33.28727604396901, 32.41822391278223 ]
        value_4 = [35.78534478322666, 40.74896955849369, 42.80249145810686, 41.96639063661255, 38.69685028772376]

        return [value_1, value_2, value_3, value_4]
    
    @staticmethod
    def define_players():
        player_vector_sizes = [1, 1, 1, 1, 1]
        player_objective_functions = [0, 1, 2, 3, 4]
        player_constraints = [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
        bounds = [(0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 75), (0, 100), (0, 150), (0, 200)] 
        bounds_training = [(0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 75), (0, 100), (0, 150), (0, 200)]
        return [player_vector_sizes, player_objective_functions, player_constraints, bounds, bounds_training]
    
    @staticmethod
    def objective_functions():
        return [A16C.obj_func_1, A16C.obj_func_2, A16C.obj_func_3, A16C.obj_func_4, A16C.obj_func_5]
    
    @staticmethod
    def objective_function_derivatives():
        return [A16C.obj_func_der_1, A16C.obj_func_der_2, A16C.obj_func_der_3, A16C.obj_func_der_4, A16C.obj_func_der_5]
    
    @staticmethod
    def constraints():
        return [A16C.g0, A16C.g1, A16C.g2, A16C.g3]
    
    @staticmethod
    def constraint_derivatives():
        return [A16C.g0_der, A16C.g1_der, A16C.g2_der, A16C.g3_der]
    
    @staticmethod
    def obj_func(
            f: npt.NDArray[np.float64],
            x: npt.NDArray[np.float64],
            c: npt.NDArray[np.float64],
            g: npt.NDArray[np.float64],
            S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        k = 5
        y = 1.1
        obj = (c * x + (g/1+g) * k ** (-1/g) * x ** ((1+g)/g) ) - 5000 ** (1/y) * x * S ** (-1/y)
        return obj
    
    @staticmethod
    def obj_func_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        S = x1 + x2 + x3 + x4 + x5
        c = 10
        g = 1.2
        return A16C.obj_func(x1, S, c, g)
    
    @staticmethod
    def obj_func_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        S = x1 + x2 + x3 + x4 + x5
        c = 8
        g = 1.1
        return A16C.obj_func(x2, S, c, g)
    
    @staticmethod
    def obj_func_3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        S = x1 + x2 + x3 + x4 + x5
        c = 6
        g = 1.0
        return A16C.obj_func(x3, S, c, g)
    
    @staticmethod
    def obj_func_4(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        S = x1 + x2 + x3 + x4 + x5
        c = 4
        g = 0.9
        return A16C.obj_func(x4, S, c, g)
    
    @staticmethod
    def obj_func_5(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        S = x1 + x2 + x3 + x4 + x5
        c = 2
        g = 0.8
        return A16C.obj_func(x5, S, c, g)
        
    @staticmethod
    def obj_func_der(
            x: npt.NDArray[np.float64],
            c: npt.NDArray[np.float64],
            g: npt.NDArray[np.float64],
            S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        
        k = 5
        y = 1.1
        obj = c + (g/1+g) * (k ** (-1/g)) * ((1+g)/g) * x ** (((1+g)/g)-1) - 5000 ** (1/y) * S ** (-1/y)
        return obj
    
    @staticmethod
    def obj_func_der_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1] #next door neighbors figthing up a storm chile
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        S = x1 + x2 + x3 + x4 + x5
        c = 10
        g = 1.2
        return A16C.obj_func_der(x1, S, c, g)
    
    @staticmethod
    def obj_func_der_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        S = x1 + x2 + x3 + x4 + x5
        c = 8
        g = 1.1
        return A16C.obj_func_der(x2, S, c, g)
    
    @staticmethod
    def obj_func_der_3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        S = x1 + x2 + x3 + x4 + x5
        c = 6
        g = 1.0
        return A16C.obj_func_der(x3, S, c, g)
    
    @staticmethod
    def obj_func_der_4(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        S = x1 + x2 + x3 + x4 + x5
        c = 4
        g = 0.9
        return A16C.obj_func_der(x4, S, c, g)
    
    @staticmethod
    def obj_func_der_5(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        S = x1 + x2 + x3 + x4 + x5
        c = 2
        g = 0.8
        return A16C.obj_func_der(x5, S, c, g)
    
    @staticmethod
    def g0(x):
        x1, x2, x3, x4, x5 = x
        return (sum(x1) + sum(x2) + sum(x3) + sum(x4) + sum(x5) - 75)[0]

    @staticmethod
    def g1(x):
        x1, x2, x3, x4, x5 = x
        return (sum(x1) + sum(x2) + sum(x3) + sum(x4) + sum(x5) - 100)[0]
    
    @staticmethod
    def g2(x):
        x1, x2, x3, x4, x5 = x
        return (sum(x1) + sum(x2) + sum(x3) + sum(x4) + sum(x5)- 150)[0]
    
    @staticmethod
    def g3(x):
        x1, x2, x3, x4, x5 = x
        return (sum(x1) + sum(x2) + sum(x3) + sum(x4) + sum(x5)- 200)[0]
    
    @staticmethod
    def g0_der(x1):
        return np.array([[1, 1, 1, 1, 1]]).reshape(-1, 1)

    @staticmethod
    def g1_der(x1):
        return np.array([[1, 1, 1, 1, 1]]).reshape(-1, 1)

    @staticmethod
    def g2_der(x1):
        return np.array([[1, 1, 1, 1, 1]]).reshape(-1, 1)

    @staticmethod
    def g3_der(x1):
        return np.array([[1, 1, 1, 1, 1]]).reshape(-1, 1)



    