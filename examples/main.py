from library import *
from library.misc import *
from problems import *


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
    problem = A10eU
    bounded = False

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
        # solver1.summary(problem.paper_solution()[0])
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

        solver1 = GNEP_Solver_Unbounded(
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
        # print('\n\n')
        solver1.summary(problem.paper_solution()[0])
        # solver1.summary()
        print('\n\n')
        print('Check NE')
        check_NE(
            sol[0].x[:sum(player_vector_sizes)],
            player_vector_sizes,
            player_objective_functions,
            problem_funcs[0],
            problem_funcs[2],
            player_constraints
        )




