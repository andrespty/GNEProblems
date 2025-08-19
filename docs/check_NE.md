## `check_NE`

### What does this function do?

The `check_NE` function is designed to **verify whether a given solution is a Nash Equilibrium (NE)** for a game. It does this by checking two main conditions:

* **Constraint satisfaction:** It first confirms that the proposed solution satisfies all the game's constraints. If any constraint is violated, the solution is immediately deemed invalid.
* **No incentive to deviate:** For each player in the game, it assumes the other players' strategies are fixed at the proposed solution's values. Then, it attempts to find a better strategy for that player using an optimization algorithm. If no better strategy exists (i.e., the player's objective function cannot be improved by a significant amount, defined by `epsilon`), it confirms that the player has no incentive to deviate from the proposed solution.

If both conditions hold for all players, the function concludes that the proposed solution is a Nash Equilibrium.

---

### Parameters

* `result` (`List[np.float64]`): The candidate solution to be verified.
* `action_sizes` (`List[int]`): A list where each integer represents the number of actions (or variables) controlled by each player.
* `objective_functions` (`List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]]`): A list of callable functions, with each function representing a player's objective.
* `player_objective_function` (`List[int]`): A list of index mapping each player to their corresponding objective function in `objective_functions`.
* `constraints` (`List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]]`): A list of callable functions, each representing a constraint of the game.
* `player_constraints` (`List[List[int]]`): A list of lists of indexes, specifying which constraints apply to which players.
* `paper_res` (`List[float]`, optional): An optional known good solution from a reliable source (e.g., a research paper) for comparison.
* `epsilon` (`float`, optional): A small tolerance value used to determine if a player has a significant incentive to deviate. The default value is `1e-3`.

---

#### How to use it?

To use `check_NE`, you must provide a candidate solution along with the complete model of your game, including player action sizes, objective functions, and all constraints.

Here's an example:

```python
import numpy as np
from scipy.optimize import minimize

# Define a simple 2-player game
# Player 1's actions are x_1, Player 2's actions are x_2
# action_sizes = [1, 1]

# Player 1's objective function: - (x_1 - 10)**2
def obj_func_1(x):
    return -(x[0] - 10)**2

# Player 2's objective function: - (x_2 - 5)**2
def obj_func_2(x):
    return -(x[1] - 5)**2

# A game with a known Nash Equilibrium at x_1 = 10 and x_2 = 5
# objective_functions = [obj_func_1, obj_func_2]

# There are no constraints in this simple example
# constraints = []
# player_constraints = [[], []]

# A candidate solution to check
# result = [10.0, 5.0]

# Call the function
check_NE(
    result=[10.0, 5.0],
    action_sizes=[1, 1],
    player_objective_function=[0, 1],
    objective_functions=[obj_func_1, obj_func_2],
    constraints=[],
    player_constraints=[[], []]
)