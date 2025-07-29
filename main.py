# from GNESolver5 import *
# from GNESolver5copy import *
from library.GNESolver6 import *
from library.GNESolverBounded import *
from Problems_Bounded.ProblemA1 import A1
from Problems_Bounded.ProblemA2 import A2
from Problems_Bounded.ProblemA3 import A3
from Problems_Bounded.ProblemA4 import A4
from Problems_Bounded.ProblemA5 import A5
from Problems_Bounded.ProblemA6 import A6
from Problems_Bounded.ProblemA7 import A7
from Problems_Bounded.ProblemA8 import A8
from Problems_Bounded.ProblemA9a import A9a
# from Problems_Bounded.ProblemA9b import A9b
from Problems_Bounded.ProblemA10a import A10a
from Problems_Bounded.ProblemA11 import A11
from Problems_Bounded.ProblemA12 import A12
from Problems_Bounded.ProblemA13 import A13
from Problems_Bounded.ProblemA14 import A14
from Problems_Bounded.ProblemA15 import A15
from Problems_Bounded.ProblemA16 import A16
from Problems_Bounded.ProblemA17 import A17
from Problems_Bounded.ProblemA18 import A18

from Problems_Unbounded.ProblemA1U import A1U # Works
from Problems_Unbounded.ProblemA2U import A2U
from Problems_Unbounded.ProblemA3U import A3U
from Problems_Unbounded.ProblemA7U import A7U

def get_problem(problem_n):
    # Define the problem
    obj = problem_n.objective_functions()
    obj_der = problem_n.objective_function_derivatives()
    c = problem_n.constraints()
    c_der = problem_n.constraint_derivatives()

    # Describe Players responsibilities
    p = problem_n.define_players()
    return [obj,obj_der,c,c_der,p]

def get_initial_point(action_sizes, player_constraints, primal_initial_point=0.01, dual_initial_point=10):
    length = len(player_constraints)
    primal = [np.reshape(np.ones(size, dtype=np.float64), [-1,1])*primal_initial_point for size in action_sizes]
    dual = [dual_initial_point for _ in range(length)]
    return primal, dual

if __name__ == '__main__':
    # Testing: Change the next line to test a problem
    problem = A15
    bounded = True

    if bounded:
        problem_funcs = get_problem(problem)
        constraints, player = problem_funcs[3:]
        (player_vector_sizes,
         player_objective_functions,
         player_constraints, bounds, bounds_training) = player
        print('Here')
        solver1 = GNEP_Solver_Bounded(
            *get_problem(problem)[:4],
            player_objective_functions,
            player_constraints,
            bounds_training,
            player_vector_sizes,
        )

        # Set Initial Point
        primal, dual = get_initial_point(
            player_vector_sizes,
            constraints,
            primal_initial_point=0.01,
            dual_initial_point=1
        )
        print(flatten_variables(primal, dual))
        # # Solve Problem
        sol = solver1.solve_game(flatten_variables(primal, dual),bounds=bounds_training )
        print('\n\n')
        solver1.summary(problem.paper_solution()[0])
        solver1.summary()
        # print(sol)
        print('\n\n')

        # solver1.nash_check()


    else:
        problem_funcs = get_problem(problem)
        constraints, player = problem_funcs[3:]
        (player_vector_sizes,
         player_objective_functions,
         player_constraints) = player

        solver1 = GNEP_Solver(
            *get_problem(problem)[:4],
            player_objective_functions,
            player_constraints,
            player_vector_sizes,
        )
        # Set Initial Point
        primal, dual = get_initial_point(player_vector_sizes, constraints, primal_initial_point=0.01, dual_initial_point=1)
        print(flatten_variables(primal, dual))
        # # Solve Problem
        sol = solver1.solve_game(flatten_variables(primal, dual))
        print('\n\n')
        solver1.summary(problem.paper_solution()[0])
        # solver1.summary()
        print('\n\n')
        solver1.nash_check()




