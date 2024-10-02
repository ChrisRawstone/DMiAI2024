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
original_cycle_durations = [30, 20, 20, 25]  # Original ticks for each index in the routine list
cycle_durations = original_cycle_durations.copy()  # Temporary cycle durations to adjust per cycle
ticks_per_group = 0  # Tracks how many ticks each group has been active

# Hyperparameter for change in the last N ticks
WINDOW_SIZE = 5  # Number of ticks over which we calculate the change

# Track signal state and timing
current_index = 0

# Initialize previous waiting vehicle counts to track changes
previous_waiting_vehicle_counts = defaultdict(int)

# History of vehicle counts for each leg (deque to store the last N ticks)
vehicle_counts_history = deque(maxlen=WINDOW_SIZE)

# History of waiting vehicle counts for each leg (deque to store the last N ticks)
waiting_counts_history = deque(maxlen=WINDOW_SIZE)

# Track if an adjustment has been made for the current index
adjustment_made = False

def run_game():
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

    global current_index, ticks_per_group, previous_waiting_vehicle_counts, vehicle_counts_history, waiting_counts_history, adjustment_made, cycle_durations

    while True:
        # Get the state from the output_queue
        state = output_queue.get()

        if state.is_terminated:
            p.join()
            break

        # Log number of vehicles in each leg and total number of vehicles
        leg_vehicle_counts = defaultdict(int)
        waiting_vehicle_counts = defaultdict(int)

        # Count vehicles and waiting vehicles (waiting if speed is 0 or near 0)
        for vehicle in state.vehicles:
            leg_vehicle_counts[vehicle.leg] += 1
            if vehicle.speed <= 0.01:  # A small threshold to consider a vehicle as waiting
                waiting_vehicle_counts[vehicle.leg] += 1

        total_vehicles = len(state.vehicles)
        # logger.info(f"Total vehicles: {total_vehicles}")

        # Calculate the change in the total number of vehicles per leg for the last WINDOW_SIZE ticks
        if len(vehicle_counts_history) == WINDOW_SIZE:
            previous_counts = vehicle_counts_history[0]  # Get counts from WINDOW_SIZE ticks ago
            change_in_vehicle_counts = {leg: leg_vehicle_counts[leg] - previous_counts.get(leg, 0) for leg in leg_vehicle_counts}

            # Adjust the green period dynamically based on the rate of change
            active_group_legs = first_list[current_index]
            active_changes = [change_in_vehicle_counts.get(leg, 0) for leg in active_group_legs]

            # Check the actual state of the signals in the active group
            for signal in state.signals:
                if signal.name in active_group_legs and signal.state == "green":
                    # Check if adjustments are necessary and if they haven't been applied already
                    if not adjustment_made:
                        if all(change < -4 for change in active_changes):
                            # logger.info(f"Prolonging green period for {active_group_legs} by 5 ticks.")
                            cycle_durations[current_index] += 5  # Temporarily adjust for the current cycle
                            adjustment_made = True
                        elif all(change > -1 for change in active_changes):
                            # logger.info(f"Reducing green period for {active_group_legs} by 6 ticks.")
                            cycle_durations[current_index] = max(1, cycle_durations[current_index] - 15)  # Ensure it doesn't go below 1 tick
                            adjustment_made = True
                        break  # Only apply the adjustment once per cycle for any green signal

            # for leg, change in change_in_vehicle_counts.items():
            #     logger.info(f"Leg {leg} has a change of {change} vehicles over the last {WINDOW_SIZE} ticks.")
        
        # Store the current tick's vehicle counts in history
        vehicle_counts_history.append(leg_vehicle_counts.copy())

        # Log change in waiting vehicles over WINDOW_SIZE ticks
        if len(waiting_counts_history) == WINDOW_SIZE:
            previous_waiting_counts = waiting_counts_history[0]  # Get waiting counts from WINDOW_SIZE ticks ago
            change_in_waiting_counts = {leg: waiting_vehicle_counts[leg] - previous_waiting_counts.get(leg, 0) for leg in waiting_vehicle_counts}
            # for leg, change in change_in_waiting_counts.items():
                # logger.info(f"Leg {leg} has a change of {change} waiting vehicles over the last {WINDOW_SIZE} ticks.")

        # Store the current tick's waiting vehicle counts in history
        waiting_counts_history.append(waiting_vehicle_counts.copy())

        # Log the current waiting vehicle counts
        # for leg, count in waiting_vehicle_counts.items():
            # logger.info(f"Leg {leg} has {count} waiting vehicles.")

        # Update previous waiting counts
        previous_waiting_vehicle_counts = waiting_vehicle_counts.copy()

        # Reset signal group if vehicle count is low
        # if total_vehicles < 5:
        #     logger.warning("Reset detected! Resetting signal groups.")
        #     current_index = 0
        #     ticks_per_group = 0
        #     cycle_durations = original_cycle_durations.copy()  # Reset cycle durations
        #     adjustment_made = False  # Allow adjustments again after reset

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
        # logger.info(f"Active signal group: {active_group}")

        next_signals = []
        for signal in state.signals:
            if signal.name in active_group:
                next_signals.append(SignalDto(name=signal.name, state="green"))
            else:
                next_signals.append(SignalDto(name=signal.name, state="red"))

        input_queue.put({signal.name: signal.state for signal in next_signals})

    if state.total_score == 0:
        state.total_score = 1e9

    inverted_score = 1. / state.total_score
    return inverted_score

if __name__ == '__main__':
    run_game()
