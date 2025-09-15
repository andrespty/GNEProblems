### i am having a bunch of issues with errors 

# Welcome to MkDocs

For full documentation visit [mkdocs.org](https://www.mkdocs.org).

## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
## one_hot_encoding

```python
def one_hot_encoding(funcs_idx: List[Union[int, PlayerConstraint]], sizes: List[int], num_functions: int) -> Matrix:
   """ 
   This function builds a matrix mapping each player’s action variables to the functions they are assigned

    Creates a zeros NumPy matrix and then iterates through functions mapping them to correct player's action variables
    and assigning them to the matrix.

    Parameters
    ----------
    funcs_idx : List
        A list of either integers or PlayerConstraint (PlayerConstraint=Union[int], None, list[None])
    sizes: List[int]
        A list of integers thats length needs to equal the length of funcs_idx
    num_functions: Int
    An integer value that indicates the number of possible functions

    Returns
    -------
    Matrix
        Returns a matrix of shape (sum(sizes), num_functions), where 
        each row represents a player's variables and each column represents
        a function.

    Examples
    --------
    >>> funcs_idx = [[0,2], None, [1]
    >>> sizes = [2,3,1]
    >> num_functions = 3
    >> M = one_hot_encoding(funcs_idx, sizes, num_functions)
    3.14
    """
```

## construct_vectors

```python
def construct_vectors(actions: Vector, action_sizes: List[int]) -> VectorList:
    """
    Split a concatenated action array into separate action vectors for each player.

    This function validates the input types and shapres, ensuring that the total
    number of rows in "actions" matches the sum of "action_sizes". It then splits
    the stacked column vector into per-player subarrays in the same order as specified
    in "action_sizes".

    Parameters
    ----------
    actions : numpy.ndarray of shape (sum(action_sizes), 1)
        A 2D NumPy array containing all players' actions stacked vertically.
        The number of rows must equal the sum of all entries in ``action_sizes``.
    action_sizes : list of int
        A list specifying the length of each player's action vector.
        The sum of these sizes must match the number of rows in ``actions``.

    Returns
    -------
    list of numpy.ndarray
        A list of 2D NumPy arrays, each corresponding to one player's action vector.
        The arrays are in the same order as the players in ``action_sizes``.

    Raises
    -------
    Type Error
        if "actions" is not a NumPy array or if "action_sizes" is not a list
        of integers

    Value Error
        If the number of rows in "actions" does not equal the sum of all
        entries in ``action_sizes``.

    Examples
    --------
    >> actions = np.array([[1.0], [2.0], [3.0], [4.0]])
    >> action_sizes = [2, 2]
    >> construct_vectors(actions, action_sizes)
    [array([[1.],
            [2.]]),
     array([[3.],
            [4.]])]
    """
```

