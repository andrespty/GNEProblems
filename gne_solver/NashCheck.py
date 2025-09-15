from gne_solver.core import get_problem
from Problems import A2U
from gne_solver.optimization import *

def nash_check(
        problem_n,
        results: List[float],
        single_obj_vector,
        paper_result:List[float]=None,
        opt_method='trust-constr', #SLSQP
        epsilon=1e-3
):
    #--------------Set up of the problem----------------
    problem = get_problem(problem_n)
    players = problem['players']
    objective_funcs = problem['obj_funcs']
    constraints = problem['constraints']
    player_vector_sizes:List[int] = players[0]
    player_objective_functions: List[int] = players[1]
    player_constraints: List[PlayerConstraint] = players[2]

    result = np.array(results).reshape(-1, 1)
    res_vector = construct_vectors(result, player_vector_sizes)

    #--------------- Values of solution ----------------
    obj_func_vals = objective_check(objective_funcs, res_vector)
    print('Computed Objective Function: \n', obj_func_vals)

    print('Checking constraints...')
    constraint_vals, constraint_sat = constraint_check(constraints, res_vector)
    print('Constraints conclusion:')
    if all(constraint_sat):
        print('Satisfied')
        # print(constraint_vals)
    else:
        print('Not Satisfied')
        failed = [i for i, sat in enumerate(constraint_sat) if not sat]
        print(f"Unsatisfied constraint IDs: {failed}")
        # print("Corresponding values:", [constraint_vals[i] for i in failed])

    print('--------------------------------------')

    conclusion = []
    # Optimize each player by fixing the opponents
    for player_idx, p_o_idx in enumerate(player_objective_functions):
        print(f"Player {player_idx + 1}")
        p_var = res_vector[player_idx]
        p_objective = objective_funcs[p_o_idx]
        p_constraints = [constraints[c_idx] for c_idx in player_constraints[p_o_idx] if c_idx]


        min_result = optimize_obj_func(
            player_idx,
            res_vector,
            p_objective,
            p_constraints,
            method=opt_method,
            single_obj_vector=single_obj_vector
        )


        #--------------Report----------------
        fixed_vars = res_vector[:player_idx] + res_vector[player_idx + 1:]
        opt_actions = np.array(min_result.x).reshape(-1, 1)
        new_vars = fixed_vars[:player_idx] + [opt_actions] + fixed_vars[player_idx:]

        opt_obj_func = objective_check(objective_funcs, new_vars)
        opt_constraint_vals, opt_constraint_sat = constraint_check(constraints, new_vars)

        print(f"Optimized Actions: {opt_actions}")
        print('Optimized Objective Functions: \n', np.array(opt_obj_func).reshape(-1, 1))
        print("Constraints conclusion:")
        if all(opt_constraint_sat):
            print('Satisfied')
        else:
            print('Not Satisfied')
            failed = [i for i, sat in enumerate(constraint_sat) if not sat]
            print(f"Unsatisfied constraint IDs: {failed}")
            print(f"Computed Solution is at the NE for Player {player_idx + 1}")
            conclusion.append(True)
            # print("Corresponding values:", [constraint_vals[i] for i in failed])
            continue

        difference = compare_solutions(
            results,
            np.concatenate(new_vars).reshape(-1).tolist(),
            player_vector_sizes,
            objective_funcs,
            solution_name=['Computed', 'Optimized']
        )
        print(f"Difference between Objective Functions of Player {player_idx + 1}:\n {difference[p_o_idx]}")
        if difference[p_o_idx] > epsilon:
            print(f'Better solution found for Player {player_idx + 1}\n Computed Solution not at the NE')
            conclusion.append(False)
        else:
            print(f"Computed Solution is at the NE for Player {player_idx + 1}")
            conclusion.append(True)

    print('Nash Equilibrium:')
    print(conclusion)
    return

# ip = [1,2,3,4,4,5,5]
# # res = [-8.03912601e-01, -3.06214541e-01, -2.35408803e+00,  9.70149229e-01,
# #   3.12283064e+00,  7.51199635e-02, -1.28107603e-01 ]
# res = [ 2.99985171e-01,  1.60834624e-01,  1.64467896e-01,  1.59598661e-01,
#   1.44111185e-01,  1.15865596e-02,  1.19428749e-02,  1.19453955e-02,
#   1.47368133e-02,  1.16354134e-02 ]
# nash_check(A2U, res, True)