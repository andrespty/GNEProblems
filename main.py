from GNESolver5 import *
from Problems.ProblemA8 import A8
from Problems.ProblemA7 import A7
from Problems.ProblemA5 import A5

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
    problem = A7

    problem_funcs = get_problem(problem)
    constraints, player = problem_funcs[3:]
    (player_vector_sizes,
     player_objective_functions,
     player_constraints,
     bounds,
     bounds_training) = player

    # Define the problem solver
    solver1 = GeneralizedNashEquilibriumSolver(
        *get_problem(problem)[:4],
        player_objective_functions,
        player_constraints,
        bounds,
        player_vector_sizes,
        useBounds=True
    )

    # Set Initial Point
    primal, dual = get_initial_point(player_vector_sizes, constraints)

    # # Solve Problem
    sol = solver1.solve_game(flatten_variables(primal, dual), bounds_training)
    print('\n\n')
    solver1.summary(problem.paper_solution()[0])
    print('\n\n')
    solver1.nash_check()




