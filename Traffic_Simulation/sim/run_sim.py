from multiprocessing import Process, Queue
from time import sleep, time
from environment import load_and_run_simulation
from collections import defaultdict
from dtos import SignalDto

# Initialize queues
input_queue = Queue()
output_queue = Queue()
error_queue = Queue()

# Time-based cycling parameters
CYCLE_DURATION = 20  # Seconds to change signal groups
OVERLAP_DURATION = 1  # Seconds of overlap between two signal groups

# List for signal combinations (cycling through these)
first_list = [
    ['A1', 'A1LeftTurn'],
    ['A2', 'A2LeftTurn'],
    ['B1', 'B1LeftTurn'],
    ['B2', 'B2LeftTurn']
]

# Track signal state
current_index = 0
previous_group = None  # To track the previous signal group for overlap
overlap_start_time = None  # To track the start time of overlap
start_time = time()

def run_game():
    test_duration_seconds = 600
    random = True
    configuration_file = "models/1/glue_configuration.yaml"
    
    # Start simulation process
    p = Process(target=load_and_run_simulation, args=(configuration_file,
                                                      time(),
                                                      test_duration_seconds,
                                                      random,
                                                      input_queue,
                                                      output_queue,
                                                      error_queue))
    p.start()

    global current_index, previous_group, overlap_start_time, start_time

    while True:
        # Get the state from the output queue
        state = output_queue.get()

        if state.is_terminated:
            p.join()
            break

        # Reset signal group if vehicle count is low
        if len(state.vehicles) < 5:
            print("Reset detected! Resetting signal groups.")
            current_index = 0
            start_time = time()

        # Time-based cycling for signal groups
        if time() - start_time >= CYCLE_DURATION:
            previous_group = first_list[current_index]  # Store the current group as the previous group
            current_index = (current_index + 1) % len(first_list)  # Cycle to the next group
            print(f"Switching to signal group: {first_list[current_index]}")
            overlap_start_time = time()  # Start overlap period
            start_time = time()  # Reset cycle timer

        # Get the active signal group
        active_group = first_list[current_index]
        print(f"Active signal group: {active_group}")

        # Set signals to green if they are in the active group or previous group during overlap
        next_signals = []
        for signal in state.signals:
            if signal.name in active_group:
                next_signals.append(SignalDto(name=signal.name, state="green"))
            elif previous_group and signal.name in previous_group and (time() - overlap_start_time) < OVERLAP_DURATION:
                # Keep previous group signals green for the overlap duration
                next_signals.append(SignalDto(name=signal.name, state="green"))
                print(f"Signal {signal.name} is still green due to overlap.")
            else:
                next_signals.append(SignalDto(name=signal.name, state="red"))

        # Print the actual signal states
        for signal in next_signals:
            print(f"Signal {signal.name} is {signal.state}.")

        # Send the next signals to the simulation input queue
        input_queue.put({signal.name: signal.state for signal in next_signals})

        # Simulate delay between cycles
        sleep(0.5)

    # End of simulation, return the score
    if state.total_score == 0:
        state.total_score = 1e9

    inverted_score = 1. / state.total_score
    return inverted_score


if __name__ == '__main__':
    run_game()
