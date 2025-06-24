# from GNESolver5 import *
from GNESolver5 import *
from library.GNESolver6 import *
# from library.GNESolver6 import *
from Problems.ProblemA1 import A1
from Problems.ProblemA2 import A2
from Problems.ProblemA3 import A3
from Problems.ProblemA4 import A4
from Problems.ProblemA5 import A5
from Problems.ProblemA6 import A6
from Problems.ProblemA7 import A7
from Problems.ProblemA8 import A8
from Problems.ProblemA11 import A11
from Problems.ProblemA12 import A12
from Problems.ProblemA13 import A13
from Problems.ProblemA17 import A17
from Problems.ProblemA18 import A18



def get_problem(problem_n):
    # Define the problem
    obj = problem_n.objective_functions()
    obj_der = problem_n.objective_function_derivatives()
    c = problem_n.constraints()
    c_der = problem_n.constraint_derivatives()

    # Describe Players responsibilities
    p = problem_n.define_players()
    return [obj,obj_der,c,c_der,p]

def get_initial_point(action_sizes, player_constraints, dual_initial_point=10):
    length = len(player_constraints)
    primal = [np.reshape(np.zeros(size, dtype=np.float64), [-1,1]) for size in action_sizes]
    dual = [dual_initial_point for _ in range(length)]
    return primal, dual

if __name__ == '__main__':
    # Testing: Change the next line to test a problem
    problem = A13

    problem_funcs = get_problem(problem)
    constraints, player = problem_funcs[3:]
    (player_vector_sizes,
     player_objective_functions,
     player_constraints,
     bounds,
     bounds_training) = player

    # Define the problem solver
    solver1 = GNEP_Solver(
        *get_problem(problem)[:4],
        player_objective_functions,
        player_constraints,
        bounds,
        player_vector_sizes,
        # useBounds=True
    )

    # Set Initial Point
    primal, dual = get_initial_point(player_vector_sizes, constraints)
    print(flatten_variables(primal, dual))
    # # Solve Problem
    sol = solver1.solve_game(flatten_variables(primal, dual), bounds_training)
    print('\n\n')
    solver1.summary(problem.paper_solution()[0])
    print('\n\n')
    solver1.nash_check()




