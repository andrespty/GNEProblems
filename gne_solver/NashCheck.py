from gne_solver.core import get_problem
from Problems import A2U
from gne_solver.optimization import *

def nash_check(
        problem_n,
        results: List[float],
        single_obj_vector,
        paper_result:List[float]=None,
        opt_method='trust-constr', #SLSQP
        epsilon=1e-2
):
    # Set up of the problem------------------------------------------------
    constraints_satisfied = False
    problem = get_problem(problem_n)
    players = problem['players']
    objective_funcs = problem['obj_funcs']
    constraints = problem['constraints']
    player_vector_sizes:List[int] = players[0]
    player_objective_functions: List[int] = players[1]
    player_constraints: List[PlayerConstraint] = players[2]

    result = np.array(results).reshape(-1, 1)
    res_vector = construct_vectors(result, player_vector_sizes)

    # Values of solution ------------------------------------------------
    obj_func_vals = objective_check(objective_funcs, res_vector)
    print('Computed Objective Function: \n', obj_func_vals)

    print('Checking constraints...')
    constraint_vals, constraint_sat = constraint_check(constraints, res_vector)
    print('Constraints conclusion:')
    if all(constraint_sat):
        print('Satisfied')
        constraints_satisfied = True
        # print(constraint_vals)
    else:
        print('Not Satisfied')
        failed = [i for i, sat in enumerate(constraint_sat) if not sat]
        print(f"Unsatisfied constraint IDs: {failed}")
        constraints_satisfied = False
        # print("Corresponding values:", [constraint_vals[i] for i in failed])

    print('--------------------------------------')

    # Fix other players and optimize each player -----------------------
    conclusion = []
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


        # Report---------------------------------------------------------------
        fixed_vars = res_vector[:player_idx] + res_vector[player_idx + 1:]
        opt_actions = np.array(min_result.x).reshape(-1, 1)
        new_vars = fixed_vars[:player_idx] + [opt_actions] + fixed_vars[player_idx:]

        opt_obj_func = objective_check(objective_funcs, new_vars)
        opt_constraint_vals, opt_constraint_sat = constraint_check(constraints, new_vars)

        # Optimized values check and constraints--------------------------------
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

        # Comparing optimized solution with algorithms solution--------------------------------
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
        print("--------------------------")

    # Conclusion of our solution------------------------------------------------
    # 1. If optimization doesn't satisfy constraints: True
    # 2. If optimization doesn't obtain a better obj function: True
    # 3. If optimization obtains better obj function: False
    print("CONCLUSION")
    print('Our solution satisfies constraints?: ', constraints_satisfied)
    if constraints_satisfied:
        print('Did we compute better solution than optimization?:')
        print(conclusion)
        if all(conclusion):
            print("Our algorithm computed the Nash Equilibrium!")
            print("Optimization could not improve our solution")

    else:
        print("Our algorithm did not compute the Nash Equilibrium")
        print("Constraints are not satisfied")
        print('Did we compute better solution than optimization?:')
        print(conclusion)

    return