from multiprocessing import Process, Queue
from time import sleep, time

from environment import load_and_run_simulation

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
    current_signal_phase = 0  # To track which signal is green
    change_duration = 5  # Duration for each signal phase in seconds
    last_change_time = time()  # Time when the last change occurred
    
    # Initialize duration tracking
    signal_durations = None
    first_iteration = True  # Flag to check if it's the first iteration


    while True:
        state = output_queue.get()

        if state.is_terminated:
            p.join()
            break
        
        # Initialize signal durations on the first iteration
        if first_iteration:
            # Create lists and dictionaries in one go
            signals = [signal.name for signal in state.signals]
            states = {signal.name: signal.state for signal in state.signals}
            
            signal_durations = {signal: 0 for signal in signals}
            
            # {signal: (state, duration)}
            signal_state_duration = {signal: (states[signal], signal_durations[signal]) for signal in signals}
            first_iteration = False
            
        # update signal state and duration without updating the lists
        for i in range(len(signals)):
            signal_state_duration[signals[i]] = (state.signals[i].state, signal_durations[signals[i]]+time()-last_change_time)
            if signal_state_duration[signals[i]][0] == 'green': #resetting time if signal is green
                signal_state_duration[signals[i]] = (signal_state_duration[signals[i]][0], 0)
            
        
        print(f'signal_state_duration: {signal_state_duration}')
        
        #setting current signal phase to green and rest to red
        next_signals = {}
        current_tick = state.simulation_ticks
        if time() - last_change_time > change_duration:
            current_signal_phase = (current_signal_phase + 1) % len(signals)  # Cycle through signals
            last_change_time = time()  # Reset the timer
            
        for idx, signal in enumerate(signals):
            if idx == current_signal_phase:
                next_signals[signal] = 'green'
            else:
                next_signals[signal] = 'red'
        
        # Send the next signals to the input queue
        input_queue.put(next_signals)

    # End of simulation, return the score
    if state.total_score == 0:
        state.total_score = 1e9

    inverted_score = 1. / state.total_score

    return inverted_score

if __name__ == '__main__':
    run_game()