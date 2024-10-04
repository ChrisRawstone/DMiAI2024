from multiprocessing import Process, Queue
from time import sleep, time

from environment import load_and_run_simulation

from sim.dtos import SignalDto
from typing import List


def get(signals: List[SignalDto], simulation_ticks: int) -> List[SignalDto]:
    """
    This function is called by the API to get the next signals to be displayed in the simulation.
    Hardcoded cycling with green ('g') and red ('r') signals.

    Parameters:
        signals (list): List of Signal objects.
        simulation_ticks (int): The current simulation tick.

    Returns:
        list: List of Signal objects.

    Usage:
        response = TrafficSimulationPredictResponseDto(
            signals=deploy_pred.get(signals, simulation_ticks)
        )

    Note:
        The "Rewritten" cycle strings are commented out and can be used to replace the manual cycle strings.
        It is written in a more readable format with multipliers to reduce the number of lines.

    """

    # Manual cycle string for Simulation 1
    # Reference: ['A1', 'A1LeftTurn', 'A2', 'A2LeftTurn', 'B1', 'B1LeftTurn', 'B2', 'B2LeftTurn']
    cycle_strings_1 = ['rrrrrgrg', 'rrrrrgrg', 'rrrrrgrg', 'rrrrrgrg', 'rrrrrgrg', 'rrrrrgrg', 'rrrrgrgr', 'rrrrgrgr', 'rrrrgrgr', 'rrrrgrgr', 'rrrrgrgr',
                       'rrrrgrgr', 'grgrrrrr', 'grgrrrrr', 'grgrrrrr', 'grgrrrrr', 'grgrrrrr', 'grgrrrrr', 'rgrgrrrr', 'rgrgrrrr', 'rgrgrrrr', 'rgrgrrrr', 'rgrgrrrr', 'rgrgrrrr',]

    # # Rewritten cycle string for Simulation 1
    # cycle_strings_1 = (
    #     ['rrrrrgrg'] * 6 +
    #     ['rrrrgrgr'] * 6 +
    #     ['grgrrrrr'] * 6 +
    #     ['rgrgrrrr'] * 6
    # )

    # Manual cycle string for Simulation 2
    # Reference: ['A1', 'A1RightTurn', 'A1LeftTurn', 'A2', 'A2RightTurn', 'A2LeftTurn', 'B1', 'B1RightTurn', 'B2']
    cycle_strings_2 = ['ggrggrrrr', 'ggrggrrrr', 'ggrggrrrr', 'ggrggrrrr', 'ggrggrrrr', 'ggrggrrrr', 'rrrgrrrrg', 'rrrgrrrrg', 'rrrgrrrrg', 'rrrrrrrrg', 'rrrrrrggg', 'rrrrrrggg', 'rrrrrrggg', 'rrrrrrggg', 'rrrrrrggg',
                       'rrrrrrggg', 'rrrrrrrrg', 'rrrgrrrrg', 'rrrgrrrrg', 'rrrgrrrrr', 'ggrggrrrr', 'ggrggrrrr', 'ggrggrrrr', 'ggrggrrrr', 'ggrggrrrr', 'ggrggrrrr', 'rrgrrgrrr', 'rrgrrgrrr', 'rrgrrgrrr', 'rrgrrgrrr', 'rrgrrgrrr', 'rrgrrgrrr',]

    # # Rewritten cycle string for Simulation 2
    # cycle_strings_2 = (
    #     ['ggrggrrrr'] * 6 +
    #     ['rrrgrrrrg'] * 3 +
    #     ['rrrrrrrrg'] * 1 +
    #     ['rrrrrrggg'] * 6 +
    #     ['rrrrrrrrg'] * 1 +
    #     ['rrrgrrrrg'] * 2 +
    #     ['rrrgrrrrr'] * 1 +
    #     ['ggrggrrrr'] * 6 +
    #     ['rrgrrgrrr'] * 6
    # )

    # Select the cycle string based on simulation detected by number of signals, default 'r'
    cycle_strings = ['rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr']

    # Simulation 1 should have 8 signals
    if len(signals) == 8:
        cycle_strings = cycle_strings_1

    # Simulation 2 should have 9 signals
    elif len(signals) == 9:
        cycle_strings = cycle_strings_2

    # Get the cycle string based on the simulation tick
    cycle_idx = simulation_ticks % len(cycle_strings)
    cycle_string = cycle_strings[cycle_idx]

    # Set the state of the signals based on the cycle string
    next_signals = []
    for i, signal in enumerate(signals):
        state = 'green' if cycle_string[i] == 'g' else 'red'
        signal.state = state
        next_signals.append(signal)

    # Return the signals
    return next_signals

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
        
        # Insert your own logic here to parse the state and 
        # select the next action to take

        print(f'Vehicles: {state.vehicles}')
        print(f'Signals: {state.signals}')

        signal_logic_errors = None
        prediction = {}
        prediction["signals"] = []
        
        # Update the desired phase of the traffic lights
        next_signals = {}
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