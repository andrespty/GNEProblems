from gne_solver import *
from gne_solver.misc import *
from problems import *
from gne_solver.utils import *

if __name__ == '__main__':
    p_obj_func = [0,1,2,3]
    p_constraints = [[1,2], None, [1,3], [None]]
    player_vector_sizes =[1,2,1,3]
    num_funcs = 4

    one = one_hot_encoding(p_obj_func, player_vector_sizes, num_funcs)
    two = one_hot_encoding(p_constraints, player_vector_sizes, num_funcs)
    print(one)
    print(two)

    # Testing: Change the next line to test a problem
    # problem_n = A16U
    # bounded = False
    #
    # problem = get_problem(problem_n)
    # (player_vector_sizes,
    #  player_objective_functions,
    #  player_constraints) = problem['players']
    #
    # solver1 = GNEP_Solver_Unbounded(
    #     problem['obj_funcs'],
    #     problem['obj_ders'],
    #     problem['constraints'],
    #     problem['constraint_ders'],
    #     player_objective_functions,
    #     player_constraints,
    #     player_vector_sizes,
    # )
    #
    # def wrapper(x):
    #     actions = construct_vectors(x[:5], player_vector_sizes)
    #     dual_actions = np.array(x[5:]).reshape(-1, 1)
    #
    #     all_objs = []
    #     for obj_func in problem['obj_funcs']:
    #         obj = obj_func(actions) - dual_actions[0] + dual_actions[1]
    #         all_objs.append(obj)
    #     grad = np.array(all_objs).reshape(-1, 1)
    #     eng = grad.T @ grad
    #
    #     c1 = -problem['constraints'][0](actions)
    #     c2 = -problem['constraints'][1](actions)
    #
    #     g1 = (dual_actions[0] ** 2 / (1 + dual_actions[0] ** 2)) * (c1 ** 2 / (1 + c1 ** 2)) + np.exp(-dual_actions[0] ** 2) * (
    #             np.maximum(0, -c1) ** 2 / (1 + np.maximum(0, -c1) ** 2))
    #     g2 = (dual_actions[1] ** 2 / (1 + dual_actions[1] ** 2)) * (c2 ** 2 / (1 + c2** 2)) + np.exp(-dual_actions[1] ** 2) * (
    #             np.maximum(0, -c2) ** 2 / (1 + np.maximum(0, -c2) ** 2))
    #
    #     g_dual = np.concatenate([g1.flatten(), g2.flatten()]).reshape(-1,1)
    #     return eng + np.sum(g_dual)
    #
    #
    # minimizer_kwargs = dict(method="L-BFGS-B")
    # result = basinhopping(
    #     wrapper,
    #     [10,10,10,10,10,10,10],
    #     stepsize=0.01,
    #     niter=1000,
    #     minimizer_kwargs=minimizer_kwargs,
    #     interval=1,
    #     niter_success=100,
    #     disp=True,
    #     # callback=stopping_criterion
    # )
    # print(result.x)
