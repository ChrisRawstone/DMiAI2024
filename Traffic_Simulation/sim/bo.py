import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from skopt.callbacks import VerboseCallback  # Import a built-in verbose callback

# Import the run_game logic from your run_sim script
from run_sim import run_game

# Define the parameter space for optimization
space = [
    Integer(10, 30, name='cycle_A1_duration'),  # Duration of A1 signal group
    Integer(10, 30, name='cycle_A2_duration'),  # Duration of A2 signal group
    Integer(10, 30, name='cycle_B1_duration'),  # Duration of B1 signal group
    Integer(10, 30, name='cycle_B2_duration'),  # Duration of B2 signal group
    Integer(5, 10, name='ProlongStep'),            # Step size for prolonging green period
    Integer(5, 20, name='ReduceStep'),             # Step size for reducing green period
    Integer(2, 8, name='When_to_Reduce')       # Time after which reduction can happen
]

# Define the objective function that we want to minimize
@use_named_args(space)
def objective(**params):
    # Extract the parameters for the signal cycle durations
    cycle_A1_duration = params['cycle_A1_duration']
    cycle_A2_duration = params['cycle_A2_duration']
    cycle_B1_duration = params['cycle_B1_duration']
    cycle_B2_duration = params['cycle_B2_duration']
    
    # Extract the ProlongStep, ReduceStep, and When_to_Reduce parameters
    ProlongStep = params['ProlongStep']
    ReduceStep = params['ReduceStep']
    When_to_Reduce = params['When_to_Reduce']

    # Construct the cycle durations list to pass to run_game
    cycle_durations = [cycle_A1_duration, cycle_A2_duration, cycle_B1_duration, cycle_B2_duration]

    # Pass the parameters to run_game and get the total score
    total_score = run_game(
        original_cycle_durations=cycle_durations,
        first_list=[['A1', 'A1LeftTurn'], ['A2', 'A2LeftTurn'], ['B1', 'B1LeftTurn'], ['B2', 'B2LeftTurn']],
        WINDOW_SIZE=5,
        ReduceStep=ReduceStep,
        ProlongStep=ProlongStep,
        When_to_Reduce=When_to_Reduce
    )

    # Since we want to mini the score
    return total_score

if __name__ == '__main__':
    # Run Bayesian Optimization
    result = gp_minimize(
        func=objective,        # Objective function to minimize
        dimensions=space,      # Parameter space
        n_calls=50,            # Number of iterations to perform
        random_state=42,        # Random state for reproducibility
        callback=[VerboseCallback(n_total=50)]  # Callback to print progress
    )

    # Print the best found parameters and the corresponding best score
    print("Best parameters found: ", result.x)
    print("Best score achieved: ", result.fun)  # Negative because we minimized, we revert it back to original score
