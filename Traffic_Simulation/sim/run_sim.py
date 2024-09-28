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

    current_signal_phase = 0  # To track which signal is green
    change_duration = 15  # Duration for each signal phase in seconds
    last_change_time = time()  # Time when the last change occurred

    # Main loop
    while True:
        state = output_queue.get()

        if state.is_terminated:
            p.join()
            break

        # Check if it's time to change signals
        if time() - last_change_time > change_duration:
            current_signal_phase = (current_signal_phase + 1) % len(state.signals)  # Cycle through signals
            last_change_time = time()  # Reset the timer

        # Set the current signal phase to green and others to red
        next_signals = {}
        for idx, signal in enumerate(state.signals):
            if idx == current_signal_phase:
                next_signals[signal.name] = 'green'
            else:
                next_signals[signal.name] = 'red'

        # Send the updated signal states to the input queue
        input_queue.put(next_signals)

        print(f"Signal states updated: {next_signals}")

    # End of simulation, return the score
    if state.total_score == 0:
        state.total_score = 1e9

    inverted_score = 1. / state.total_score

    return inverted_score

if __name__ == '__main__':
    run_game()
