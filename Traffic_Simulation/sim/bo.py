import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from skopt.callbacks import VerboseCallback  # Import a built-in verbose callback
from concurrent.futures import ThreadPoolExecutor  # Use threading instead of multiprocessing

# Import the run_game logic from your run_sim script
from run_sim import run_game

# Define the parameter space for optimization
space = [
    Integer(15, 30, name='cycle_A1_duration'),  # Duration of A1 signal group
    Integer(5, 20, name='cycle_A2_duration'),  # Duration of A2 signal group
    Integer(5, 20, name='cycle_B1_duration'),  # Duration of B1 signal group
    Integer(5, 20, name='cycle_B2_duration'),  # Duration of B2 signal group
    # Integer(5,5, name='ProlongStep'),         # Step size for prolonging green period
    Integer(5, 15, name='ReduceStep'),          # Step size for reducing green period
    # Integer(4,4, name='When_to_Reduce')        # Time after which reduction can happen
]

# Define the objective function with threading
@use_named_args(space)
def objective(**params):
    # Extract the parameters for the signal cycle durations
    cycle_A1_duration = params['cycle_A1_duration']
    cycle_A2_duration = params['cycle_A2_duration']
    cycle_B1_duration = params['cycle_B1_duration']
    cycle_B2_duration = params['cycle_B2_duration']
    
    # Extract the ProlongStep, ReduceStep, and When_to_Reduce parameters
    # ProlongStep = params['ProlongStep']
    ProlongStep = 5
    ReduceStep = params['ReduceStep']
    # When_to_Reduce = params['When_to_Reduce']
    When_to_Reduce = 4

    # Construct the cycle durations list to pass to run_game
    cycle_durations = [cycle_A1_duration, cycle_A2_duration, cycle_B1_duration, cycle_B2_duration]

    # Use a ThreadPoolExecutor to run run_game in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Use submit to run the game asynchronously
        future = executor.submit(
            run_game,
            cycle_durations,
            [['A1', 'A1LeftTurn'], ['A2', 'A2LeftTurn'], ['B1', 'B1LeftTurn'], ['B2', 'B2LeftTurn']],  # first_list
            5,  # WINDOW_SIZE
            ReduceStep,
            ProlongStep,
            When_to_Reduce
        )

        # Get the result from the future
        total_score = future.result()

    # Since we want to minimize the score, we return it as-is
    return total_score

if __name__ == '__main__':
    # Run Bayesian Optimization
    result = gp_minimize(
        func=objective,        # Objective function to minimize
        dimensions=space,      # Parameter space
        n_calls=50,            # Number of iterations to perform
        random_state=42,       # Random state for reproducibility
        callback=[VerboseCallback(n_total=50)]  # Callback to print progress
    )

    # Print the best found parameters and the corresponding best score
    print("Best parameters found: ", result.x)
    print("Best score achieved: ", result.fun)
