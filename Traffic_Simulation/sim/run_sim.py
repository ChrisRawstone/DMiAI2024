from multiprocessing import Process, Queue
from environment import load_and_run_simulation
from collections import defaultdict, deque
from dtos import SignalDto

# Initialize queues
input_queue = Queue()
output_queue = Queue()
error_queue = Queue()

# List for signal combinations (cycling through these)
first_list = [
    ['A1', 'A1LeftTurn'],
    ['A2', 'A2LeftTurn'],
    ['B1', 'B1LeftTurn'],
    ['B2', 'B2LeftTurn']
]


# These variables shall be used for bayesian optimization
# Define original durations in ticks for each index
original_cycle_durations = [30, 30, 25, 25]
ticks_reduction = 15  # Ticks of duration reduction


cycle_durations = original_cycle_durations.copy()
ticks_per_group = 0  # Tracks how many ticks each group has been active
current_index = 0  # Tracks which signal group is currently active
duration_reduced = False  # Boolean flag to ensure reduction happens only once per cycle



# Track vehicle counts for each leg for the last 5 ticks
vehicle_counts_history = defaultdict(lambda: deque(maxlen=5))

# Logging variables for signal state durations
signal_state_durations = defaultdict(lambda: defaultdict(int))

# Function to log results
def log_results_to_file():
    filename = f"traffic_log_sim.txt"
    with open(filename, "w") as f:
        f.write(f"Results at tick {ticks_per_group}:\n")
        f.write("Signal state durations (in ticks):\n")
        for signal, states in signal_state_durations.items():
            f.write(f"Signal {signal}: {states}\n")

# Main simulation function with temporary duration adjustments
def run_game():
    test_duration_seconds = 300
    random = False
    configuration_file = "models/1/glue_configuration.yaml"

    # Start simulation process
    p = Process(target=load_and_run_simulation, args=(configuration_file, None, test_duration_seconds, random, input_queue, output_queue, error_queue))
    p.start()

    global current_index, ticks_per_group, duration_reduced
    while True:
        # Get the state from the output_queue
        state = output_queue.get()
        if state.is_terminated:
            p.join()
            break

        # Extract legs dynamically
        legs = [state.legs[leg].name for leg in range(len(state.legs))]
        current_leg = legs[current_index]  # Only one active leg

        # Count the number of vehicles in the active leg
        leg_vehicle_count = 0
        for vehicle in state.vehicles:
            if vehicle.leg == current_leg:
                leg_vehicle_count += 1

        # Store the current vehicle count in the history for the active leg
        vehicle_counts_history[current_leg].append(leg_vehicle_count)

        # Check if we have 5 ticks of history for the active leg
        if len(vehicle_counts_history[current_leg]) == 5:
            # Calculate the change in vehicle count over the last 5 ticks
            change_in_vehicle_count = vehicle_counts_history[current_leg][-1] - vehicle_counts_history[current_leg][0]
            # print(f"current leg: {current_leg}, vehicle count: {leg_vehicle_count}, change in vehicle count: {change_in_vehicle_count}")
            # Temporarily adjust the current cycle duration if the change is greater than -2 and hasn't been reduced yet
            temporary_duration = cycle_durations[current_index]
            if change_in_vehicle_count > -1 and not duration_reduced:
                temporary_duration = max(1, temporary_duration - ticks_reduction)  # Ensure duration doesn't go below 1
                duration_reduced = True  # Set flag to prevent further reduction this cycle
                print(f"Reducing duration for {current_leg} to {temporary_duration} ticks")
        else:
            temporary_duration = cycle_durations[current_index]

        ticks_per_group += 1

        # Check if the current signal group's (temporary) duration has been reached
        if ticks_per_group >= temporary_duration:
            # Move to the next signal group
            current_index = (current_index + 1) % len(first_list)
            ticks_per_group = 0  # Reset tick counter for the new signal group
            duration_reduced = False  # Reset the flag for the new cycle

        # Set signal states for the next tick based on the current signal group
        active_group = first_list[current_index]
        next_signals = []
        for signal in state.signals:
            if signal.name in active_group:
                next_signals.append(SignalDto(name=signal.name, state="green"))
                signal_state_durations[signal.name]["green"] += 1  # Log green state duration
            else:
                next_signals.append(SignalDto(name=signal.name, state="red"))
                signal_state_durations[signal.name]["red"] += 1  # Log red state duration

        # Send the next signal states back to the simulation
        input_queue.put({signal.name: signal.state for signal in next_signals})

        # Log results to the file after each tick
        log_results_to_file()

    return state.total_score

if __name__ == '__main__':
    run_game()
