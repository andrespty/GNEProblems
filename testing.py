from GNESolver5 import *
from Problems_Bounded.ProblemA1 import A1
from Problems_Bounded.ProblemA2 import A2
from Problems_Bounded.ProblemA3 import A3
from Problems_Bounded.ProblemA8 import A8
from Problems_Bounded.ProblemA7 import A7
from Problems_Bounded.ProblemA5 import A5
from Problems_Bounded.ProblemA11 import A11
from Problems_Bounded.ProblemA10a import A10a
from library.misc import *
from library.GNESolverUnbounded import *

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
    P = 3
    F = 2
    C = 5
    N = F + C + 1
    # F_players = [1 for _ in range(F*P)]
    F_players = [1,1,1,1.5,1.5,1.5]
    # C_players = [2 for _ in range(C*P)]
    C_players = [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5]
    M_player = [3 for _ in range(P)]
    players = np.array(F_players + C_players + M_player).reshape(-1,1)
    player_vector_sizes = [P for _ in range(N)]
    # print(construct_vectors(players, player_vector_sizes))
    x = construct_vectors(players, player_vector_sizes)
    print("Vectorized actions: ",x)

    p = x[-1].reshape(-1, 1)
    y_i = np.hstack(x[:F]).reshape(P, -1)
    x_i = np.hstack(x[F:F+C]).reshape(P, -1)
    print("Market actions:",p)
    print("F Player actions:",y_i)
    print("C Player actions:",x_i)
    x_12 = np.hstack(x[F:F+C][:2]).reshape(P, -1)
    print("C_12 Player actions:",x_12)

    func_firms = A10a.obj_func_firms(x)
    func_market = A10a.obj_func_market(x)
    func_consumer1 = A10a.obj_func_consumers_1(x)

    print('Func firms: ',func_firms)
    print("func market: ",func_market)
    print("func consumer1: ",func_consumer1)

    constraints = A10a.g1_der(players)
    print('constraints: ',constraints)

    # problem_funcs = get_problem(problem)
    # constraints, player = problem_funcs[3:]
    # (player_vector_sizes,
    #  player_objective_functions,
    #  player_constraints) = player
    # # Define the problem solver
    #
    # solver1 = GNEP_Solver(
    #     *get_problem(problem)[:4],
    #     player_objective_functions,
    #     player_constraints,
    #     player_vector_sizes,
    # )
    # primal, dual = get_initial_point(player_vector_sizes, constraints, dual_initial_point=0)
    #
    # print(primal, dual)
    # player_vars = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    # constraint_vars = [0, 0, 0, 0, 0, 0, 0]
    # ip = player_vars + constraint_vars
    # print(solver1.wrapper(ip))
