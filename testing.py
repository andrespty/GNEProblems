from GNESolver5 import *
from Problems_Bounded.ProblemA1 import A1
from Problems_Bounded.ProblemA2 import A2
from Problems_Bounded.ProblemA3 import A3
from Problems_Bounded.ProblemA8 import A8
from Problems_Bounded.ProblemA7 import A7
from Problems_Bounded.ProblemA5 import A5
from Problems_Bounded.ProblemA11 import A11
from library.misc import *
from library.GNESolver6 import *

from Problems_Unbounded.ProblemA1U import A1U
from Problems_Unbounded.ProblemA2U import A2U

def get_problem(problem_n):
    # Define the problem
    obj = problem_n.objective_functions()
    obj_der = problem_n.objective_function_derivatives()
    c = problem_n.constraints()
    c_der = problem_n.constraint_derivatives()

    # Describe Players responsibilities
    p = problem_n.define_players()
    return [obj,obj_der,c,c_der,p]

def get_initial_point(action_sizes, player_constraints, primal_ip=0.01, dual_initial_point=10):
    length = len(player_constraints)
    primal = [np.reshape(np.ones(size, dtype=np.float64) * primal_ip , [-1,1]) for size in action_sizes]
    dual = [dual_initial_point for _ in range(length)]
    return primal, dual


if __name__ == '__main__':
    problem = A2U
    problem_funcs = get_problem(problem)
    constraints, player = problem_funcs[3:]
    (player_vector_sizes,
     player_objective_functions,
     player_constraints) = player
    # Define the problem solver

    solver1 = GNEP_Solver(
        *get_problem(problem)[:4],
        player_objective_functions,
        player_constraints,
        player_vector_sizes,
    )
    primal, dual = get_initial_point(player_vector_sizes, constraints, dual_initial_point=0)

    print(primal, dual)
    player_vars = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    constraint_vars = [0, 0, 0, 0, 0, 0, 0]
    ip = player_vars + constraint_vars
    print(solver1.wrapper(ip))
