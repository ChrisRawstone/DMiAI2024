from multiprocessing import Process, Queue
from time import sleep, time
from environment import load_and_run_simulation
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import numpy as np
from collections import defaultdict, deque
from loguru import logger

# Import necessary functions
from pareto import (
    extract_data_for_optimization,
    estimate_acceleration_by_distance,
    estimate_queue_length,
    estimate_arrivals,
    estimate_departures,
    calculate_payoff,
    compute_pareto_solution
)

# Define the parameter space based on the ranges provided
from skopt.space import Space
from skopt.space import Categorical, Real

space = [
    # Integer values with larger step size
    Real(4, 10, name='delta_t'),   # 5-24 with step size 2
    Real(0.6, 2.0, name='saturation_flow_rate'),  # 0.6-2.0 with step size 0.2
    Real(3, 6, name='yellow_time'),         # 2,3,4 with step size 1
    Real(6, 15, name='min_green_time'), # 6-15 with step size 3
    Categorical([15, 20, 25, 30, 35, 40], name='max_green_time'),  # 30-50 with step size 5
    Real(0.51, 0.89, name='stay_will'),  # 0.5-0.9 with step size 0.1
]

# space = [
#     Real(4., 6., name = 'yellow_time'),
#     Real(1.4, 1.6, name = 'saturation_flow_rate'),
#     Real(4.9, 6.1, name = 'delta_t'),
# ]

# Function to log parameters and scores to a file
def log_to_file(parameters, score, std, filename="bo_simulation_log.txt"):
    with open(filename, "a") as f:
        f.write(f"Parameters: {parameters}\n")
        f.write(f"Total score: {score}\n")
        f.write(f"Standard Deviation: {std}\n")
        f.write("="*40 + "\n")

# Adjust the run_game function to accept parameters
def run_game(delta_t, saturation_flow_rate, yellow_time, min_green_time, max_green_time, stay_will):
    test_duration_seconds = 300
    random = True
    configuration_file = "models/1/glue_configuration.yaml"
    start_time = time()

    input_queue = Queue()
    output_queue = Queue()
    error_queue = Queue()

    p = Process(target=load_and_run_simulation, args=(configuration_file,
                                                      start_time,
                                                      test_duration_seconds,
                                                      random,
                                                      input_queue,
                                                      output_queue,
                                                      error_queue))

    p.start()

    # Wait for the simulation to start
    sleep(0.2)

    num_phases = 4  # Number of phases in the signal cycle
    current_phase = 0  # Start with phase P1

    # Define the signal groups for each phase
    phase_signals = {
        0: ['A1', 'A1LeftTurn'],  # Phase 1
        1: ['A2', 'A2LeftTurn'],  # Phase 2
        2: ['B1', 'B1LeftTurn'],  # Phase 3
        3: ['B2', 'B2LeftTurn']   # Phase 4
    }

    green_durations = [0] * num_phases  # Track how long each phase has been green

    while True:
        state = output_queue.get()

        if state.is_terminated:
            p.join()
            break

        # Extract vehicle speeds, distances, and accelerations per leg
        leg_data = extract_data_for_optimization(state)

        # Call the Pareto solution to decide whether to stay or switch phase
        next_phase, should_switch = compute_pareto_solution(leg_data,
                                                            current_phase,
                                                            delta_t,
                                                            saturation_flow_rate,
                                                            yellow_time,
                                                            min_green_time,
                                                            max_green_time,
                                                            green_durations,
                                                            stay_will)

        if should_switch:
            current_phase = next_phase
            green_durations = [0] * num_phases  # Reset all green durations when switching
        else:
            green_durations[current_phase] += 1  # Increment time for the current green phase

        # Prepare the response with the correct signals for the new phase
        green_signal_group = phase_signals[current_phase]  # Get the signal group for the current phase
        all_signals = state.signal_groups  # List of all signal groups

        # Set the signals to green for the current phase and red for others
        response_signals = {}
        for signal_group in all_signals:
            signal_state = "green" if signal_group in green_signal_group else "red"
            response_signals[signal_group] = signal_state

        # Send the updated signal data to SUMO
        input_queue.put(response_signals)

    # Print and return the total score
    total_score = state.total_score
    print(f"Total score: {total_score}")
    return total_score

# Define the objective function for Bayesian Optimization
@use_named_args(space)
def objective(**params):
    delta_t = params['delta_t']
    saturation_flow_rate = params['saturation_flow_rate']
    yellow_time = params['yellow_time']
    min_green_time = params['min_green_time']
    max_green_time = params['max_green_time']
    stay_will = params['stay_will']

    
    # min_green_time = 6
    # max_green_time = 30
    # stay_will = 0.6


    # Log the used parameters
    print(f"Running simulation with parameters:")
    print(f"delta_t={delta_t}, saturation_flow_rate={saturation_flow_rate}, yellow_time={yellow_time},")
    print(f"min_green_time={min_green_time}, max_green_time={max_green_time}, stay_will={stay_will}")

    # Run the simulation and get the total score
    scores = [run_game(delta_t, saturation_flow_rate, yellow_time, min_green_time, max_green_time, stay_will) for _ in range(12)]
    total_score = np.mean(scores)
    total_std = np.std(scores)
    
    logger.info(f"\033[93mAverage Total Score: {total_score}\033[0m")
    logger.info(f"\033[93mStandard Deviation: {total_std}\033[0m")
    # Log to file
    
    log_to_file(parameters=params, score=total_score, std=total_std)

    # Return the total score (we aim to minimize this)
    return total_score

if __name__ == '__main__':
    # Run Bayesian Optimization
    res = gp_minimize(objective,
                      space,
                      n_calls=300,          # Number of evaluations of the objective function
                      n_random_starts=2,   # Number of random initialization points
                      random_state=0)

    # Print the best result
    print(f"Best total score: {res.fun}")
    print("Best parameters:")
    for name, val in zip([dim.name for dim in space], res.x):
        print(f"{name} = {val}")
