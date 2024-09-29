from multiprocessing import Process, Queue
from time import sleep, time

from environment import load_and_run_simulation

import pprint
import inspect

from multiprocessing import Process, Queue
from time import sleep, time
from stable_baselines3 import PPO
import numpy as np
from environment import load_and_run_simulation
from trafficenv import TrafficEnv

def run_game():

    test_duration_seconds = 600
    random = True
    configuration_file = "models/1/glue_configuration.yaml"
    start_time = time()

    input_queue = Queue()
    output_queue = Queue()
    error_queue = Queue()
    errors = []

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

    # For logging
    actions = {}

    while True:

        state = output_queue.get()

        if state.is_terminated:
            p.join()
            break
        
        elapsed_time = time() - start_time
        # next_signals = {}

        # Insert your own logic here to parse the state and 
        # select the next action to take

        signal_logic_errors = None
        prediction = {}
        prediction["signals"] = []
        
        # Update the desired phase of the traffic lights
        next_signals = {}


        # next_signals["B2"]="green"


        print(f"Elapsed time: {elapsed_time}")
        print("B2:",state.signals[6])

        if elapsed_time < 10:
            next_signals["A1"] = "green"
        else:
            print("Changing to red")
            next_signals["A1"] = "red"

        current_tick = state.simulation_ticks

        for signal in prediction['signals']:
            actions[current_tick] = (signal['name'], signal['state'])
            next_signals[signal['name']] = signal['state']

        signal_logic_errors = input_queue.put(next_signals)

        if signal_logic_errors:
            errors.append(signal_logic_errors)

    # End of simulation, return the score

    # Transform the score to the range [0, 1]
    if state.total_score == 0:
        state.total_score = 1e9

    inverted_score = 1. / state.total_score


    return inverted_score

def run_trained_model():
    test_duration_seconds = 600
    random = True
    configuration_file = "models/1/glue_configuration.yaml"
    start_time = time()

    # Queues for communication between the simulation process and this script
    input_queue = Queue()
    output_queue = Queue()
    error_queue = Queue()

    # Launch the simulation in a separate process
    p = Process(target=load_and_run_simulation, args=(configuration_file, start_time, test_duration_seconds, random, input_queue, output_queue, error_queue))
    p.start()

    # Wait for the simulation to initialize
    sleep(5)

    # Load the trained PPO model
    model = PPO.load("logs/best_model/best_model.zip")

    # Initialize the environment (ensure TrafficEnv is working properly)
    env = TrafficEnv(input_queue, output_queue, configuration_file)

    actions = {}
    errors = []
    start_time = time()

    # Continue until the simulation terminates
    while True:
        # Get the current state from the simulation
        state = output_queue.get()

        # Check if the simulation is terminated
        if state.is_terminated:
            p.join()
            break

        # Get the current observation from the environment (state vector)
        observation = env._extract_state(state)

        # Use the trained PPO model to predict the next action
        action, _ = model.predict(observation)

        # Map the action back to the corresponding signal combinations
        next_signals = {signal.name: 'red' for signal in state.signals}
        selected_combination = env.actions[action]
        for signal in selected_combination:
            next_signals[signal] = 'green'

        # Send the action to the simulation
        input_queue.put(next_signals)

        # Log the action and signals for debugging purposes
        current_tick = state.simulation_ticks
        actions[current_tick] = next_signals

        # Print current state information for debugging
        elapsed_time = time() - start_time
        # print(f"Elapsed time: {elapsed_time:.2f} seconds, current tick: {current_tick}")
        # print(f"Next signals: {next_signals}")

        # Check for any signal logic errors
        signal_logic_errors = error_queue.get() if not error_queue.empty() else None
        if signal_logic_errors:
            errors.append(signal_logic_errors)

    # End of simulation, return the total score
    if state.total_score == 0:
        state.total_score = 1e9

    inverted_score = 1. / state.total_score
    print(f"Final score: {inverted_score}")

    return inverted_score

if __name__ == '__main__':
    # run_game()
    run_trained_model()