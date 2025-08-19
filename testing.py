from library import check_NE, get_problem
from problems import *

if __name__ == '__main__':
    problem_n = A2U

    problem = get_problem(problem_n)
    (player_vector_sizes,
     player_objective_functions,
     player_constraints) = problem['players']

    value_1 = [0.29962894677774, 0.00997828224734, 0.00997828224734,
               0.00997828224734, 0.59852469355630, 0.02187270661760,
               0.00999093169361, 0.00999093169361, 0.00999093169361,
               0.00999093169361]
    print(check_NE(
        value_1,
        player_vector_sizes,
        player_objective_functions,
        problem['obj_funcs'],
        problem['constraints'],
        player_constraints,
        single_obj_vector=True
    ))