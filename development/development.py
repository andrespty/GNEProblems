from gne_solver.GNESolverUnbounded import *
from gne_solver.misc import *
# from Problems_Bounded.ProblemA8 import A8
# from ProblemA8_BL import A8_BL


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
    problem = A2_BL

    problem_funcs = get_problem(problem)
    constraints_der, player = problem_funcs[3:]
    (player_vector_sizes,
     player_objective_functions,
     player_constraints,
     bounds,
     bounds_training) = player
    # Define the problem solver
    """
        GNESolverBoundless requires:
        obj_funcs:                      list of functions
        derivative_obj_funcs:           list of functions
        constraints:                    list of functions
        derivative_constraints:         list of functions
        
        player_obj_func:                list of indexes
        player_constraints:             list of list of indexes [[0,1], [0,2],...,[None]]
        bounds:                         list of tuples
        player_vector_sizes:            list of numbers
    """
    solver1 = GNEP_Solver(
        *get_problem(problem)[:4],
        player_objective_functions,
        player_constraints,
        bounds,
        player_vector_sizes
    )

    # # Set Initial Point
    primal, dual = get_initial_point(player_vector_sizes, constraints_der)
    ip = flatten_variables(primal, dual)
    print('Initial Guess: ',ip)
    solver1.wrapper(ip)
    # # # Solve Problem
    # sol = solver1.solve_game(flatten_variables(primal, dual), bounds_training)
    # print('\n\n')
    # solver1.summary(problem.paper_solution()[0])
    # print('\n\n')
    # solver1.nash_check()



