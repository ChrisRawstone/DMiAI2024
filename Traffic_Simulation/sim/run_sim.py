from multiprocessing import Process, Queue
from environment import load_and_run_simulation
from collections import defaultdict, deque
from dtos import SignalDto
from loguru import logger  # Import the logger

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

# Define original durations in ticks for each index
original_cycle_durations = [30, 15, 15, 25]  # Original ticks for each index in the routine list
# Hyperparameter for change in the last N ticks
WINDOW_SIZE = 5  # Number of ticks over which we calculate the change
ReduceStep = 15
ProlongStep = 5
When_to_Reduce = 4


cycle_durations = original_cycle_durations.copy()  # Temporary cycle durations to adjust per cycle
ticks_per_group = 0  # Tracks how many ticks each group has been active
# Track signal state and timing
current_index = 0
# Initialize previous vehicle counts to track changes
previous_vehicle_counts = defaultdict(int)
# History of vehicle counts for each leg (deque to store the last N ticks)
vehicle_counts_history = deque(maxlen=WINDOW_SIZE)
# Track if an adjustment has been made for the current index
adjustment_made = False

def run_game(original_cycle_durations, first_list, WINDOW_SIZE, ReduceStep, ProlongStep, When_to_Reduce):
    test_duration_seconds = 300
    random = True
    configuration_file = "models/1/glue_configuration.yaml"
    
    # Start simulation process
    p = Process(target=load_and_run_simulation, args=(configuration_file,
                                                      None,
                                                      test_duration_seconds,
                                                      random,
                                                      input_queue,
                                                      output_queue,
                                                      error_queue))
    p.start()

    global current_index, ticks_per_group, previous_vehicle_counts, vehicle_counts_history, adjustment_made, cycle_durations

    while True:
        # Get the state from the output_queue
        state = output_queue.get()

        if state.is_terminated:
            p.join()
            break

        # Log number of vehicles in each leg and total number of vehicles
        leg_vehicle_counts = defaultdict(int)

        # Count vehicles in each leg
        for vehicle in state.vehicles:
            leg_vehicle_counts[vehicle.leg] += 1

        # Calculate the change in the total number of vehicles per leg for the last WINDOW_SIZE ticks
        if len(vehicle_counts_history) == WINDOW_SIZE:
            previous_counts = vehicle_counts_history[0]  # Get counts from WINDOW_SIZE ticks ago
            change_in_vehicle_counts = {leg: leg_vehicle_counts[leg] - previous_counts.get(leg, 0) for leg in leg_vehicle_counts}

            # Adjust the green period dynamically based on the rate of change in total vehicles
            active_group_legs = first_list[current_index]
            active_changes = [change_in_vehicle_counts.get(leg, 0) for leg in active_group_legs]

            # Check the actual state of the signals in the active group
            if ticks_per_group > When_to_Reduce:
                for signal in state.signals:
                    if signal.name in active_group_legs and signal.state == "green":
                        # Check if adjustments are necessary and if they haven't been applied already
                        if not adjustment_made:
                            if all(change < -4 for change in active_changes):
                                # Prolong green period if there is a significant decrease in vehicles
                                cycle_durations[current_index] += ProlongStep  # Temporarily adjust for the current cycle
                                # logger.info(f"Prolonging green period for {active_group_legs} by {ProlongStep} ticks.")
                                adjustment_made = True
                            elif all(change > -1 for change in active_changes):
                                # Reduce green period if there is an increase in vehicles
                                cycle_durations[current_index] = max(1, cycle_durations[current_index] - ReduceStep)  # Ensure it doesn't go below 1 tick
                                # logger.info(f"Reducing green period for {active_group_legs} by {ReduceStep} ticks.")
                                adjustment_made = True
                            break  # Only apply the adjustment once per cycle for any green signal

        # Store the current tick's vehicle counts in history
        vehicle_counts_history.append(leg_vehicle_counts.copy())

        # Increment tick count for current group
        ticks_per_group += 1

        # Check if cycle duration has passed
        if ticks_per_group >= cycle_durations[current_index]:
            current_index = (current_index + 1) % len(first_list)
            # logger.info(f"Switching to signal group: {first_list[current_index]}")
            ticks_per_group = 0
            adjustment_made = False  # Reset adjustment flag for the new group
            cycle_durations = original_cycle_durations.copy()  # Reset cycle durations for the new group

        # Get active signal group and set signal states
        active_group = first_list[current_index]
        next_signals = []
        for signal in state.signals:
            if signal.name in active_group:
                next_signals.append(SignalDto(name=signal.name, state="green"))
            else:
                next_signals.append(SignalDto(name=signal.name, state="red"))

        input_queue.put({signal.name: signal.state for signal in next_signals})

    return state.total_score

if __name__ == '__main__':
    run_game()
