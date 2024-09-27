from multiprocessing import Process, Queue
from time import sleep, time

from environment import load_and_run_simulation

import pprint
import inspect

def print_all_variables_and_attributes(variables):
    for var_name, var_value in variables.items():
        print(f"\nVariable: {var_name} ({type(var_value).__name__})")
        
        # Check if it's a built-in type or user-defined object
        if isinstance(var_value, (int, float, str, list, dict, tuple, bool)):
            pprint.pprint(var_value)
        elif hasattr(var_value, '__dict__'):
            # Print attributes of user-defined objects
            pprint.pprint(vars(var_value))
        else:
            print("Attributes:")
            attributes = dir(var_value)
            for attr in attributes:
                if not attr.startswith('__') and not inspect.ismethod(getattr(var_value, attr)):
                    print(f"  {attr} = {getattr(var_value, attr)}")

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

        # print(f'Vehicles: {state.vehicles}')
        # print(f'Signals: {state.signals}')

        signal_logic_errors = None
        prediction = {}
        prediction["signals"] = []
        
        # Update the desired phase of the traffic lights
        next_signals = {}


        # next_signals["B2"]="green"


        print(f"Elapsed time: {elapsed_time}")
        print("B2:",state.signals[6])

        if elapsed_time < 5:
            next_signals["B2"] = "green"
        else:
            print("Changing to red")
            next_signals["B2"] = "red"

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

if __name__ == '__main__':
    run_game()