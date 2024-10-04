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

# Define original durations in ticks for each index
original_cycle_durations = [30, 25, 25, 25]
cycle_durations = original_cycle_durations.copy()
ticks_per_group = 0  # Tracks how many ticks each group has been active

# Adaptive timing constants
WINDOW_SIZE = 5  # Number of ticks over which we calculate the change
PROLONG_TICKS = 0  # Prolong the green period by 5 ticks if waiting cars decrease significantly
REDUCE_TICKS = 0  # Fallback full reduction value if needed
speed_threshold = 0.5  # Speed threshold to consider a vehicle as waiting

# Hyperparameters
ReduceStep = 15
ProlongStep = 5
When_to_Reduce = 2
current_index = 0

# Logging variables for signal state durations and active group durations
signal_state_durations = defaultdict(lambda: defaultdict(int))
active_group_durations = defaultdict(int)

# Variables to track previous waiting vehicle counts for adaptive adjustments
previous_waiting_vehicle_counts = defaultdict(int)
vehicle_counts_history = deque(maxlen=WINDOW_SIZE)
adjustment_made = False

# Function to log results
def log_results_to_file():
    filename = f"traffic_log_sim.txt"
    with open(filename, "w") as f:
        f.write(f"Results at tick {ticks_per_group}:\n")
        f.write("Signal state durations (in ticks):\n")
        for signal, states in signal_state_durations.items():
            f.write(f"Signal {signal}: {states}\n")
        f.write("\nActive group durations (in ticks):\n")
        for signal, ticks in active_group_durations.items():
            f.write(f"Signal {signal}: {ticks} ticks in active group\n")

# Main simulation function
def run_game():
    test_duration_seconds = 300
    random = True
    configuration_file = "models/1/glue_configuration.yaml"
    
    # Start simulation process
    p = Process(target=load_and_run_simulation, args=(configuration_file, None, test_duration_seconds, random, input_queue, output_queue, error_queue))
    p.start()
    global current_index, ticks_per_group, vehicle_counts_history, adjustment_made, cycle_durations
    while True:
        # print("cycle_durations:   ", cycle_durations)
        # Get the state from the output_queue
        state = output_queue.get()
        if state.is_terminated:
            p.join()
            break
        
        # Log number of vehicles in each leg
        leg_vehicle_counts = defaultdict(int)
        waiting_vehicle_counts = defaultdict(int)
        for vehicle in state.vehicles:
            leg_vehicle_counts[vehicle.leg] += 1
            if vehicle.speed <= speed_threshold:
                waiting_vehicle_counts[vehicle.leg] += 1

        # Store the current tick's waiting vehicle counts in history
        vehicle_counts_history.append(waiting_vehicle_counts.copy())
    
        # Calculate the change in waiting vehicle counts for adaptive timing
        if len(vehicle_counts_history) == WINDOW_SIZE and ticks_per_group > When_to_Reduce:
            previous_counts = vehicle_counts_history[0]
            change_in_vehicle_counts = {leg: waiting_vehicle_counts[leg] - previous_counts.get(leg, 0) for leg in waiting_vehicle_counts}
            active_group_legs = first_list[current_index]
            active_changes = [change_in_vehicle_counts.get(leg, 0) for leg in active_group_legs]
            
            if all(change < -4 for change in active_changes) and not adjustment_made:
                cycle_durations[current_index] += ProlongStep
                adjustment_made = True
                # print(f"Prolonging green period for {active_group_legs} by {PROLONG_TICKS} ticks.")
            elif all(change > -1 for change in active_changes) and not adjustment_made:
                cycle_durations[current_index] = max(1, cycle_durations[current_index] - ReduceStep)
                adjustment_made = True
        
        ticks_per_group += 1
        
        # Adjust green time based on the number of waiting vehicles
        total_waiting_vehicles = sum(waiting_vehicle_counts.values())

        if total_waiting_vehicles > 0:  # Avoid division by zero
            for leg in active_group_legs:
                leg_weight = waiting_vehicle_counts.get(leg, 0) / total_waiting_vehicles
                # Increase cycle duration proportionally
                cycle_durations[current_index] += int(ProlongStep * leg_weight)


        # Check if cycle duration has passed
        if ticks_per_group >= cycle_durations[current_index]:
            current_index = (current_index + 1) % len(first_list)
            ticks_per_group = 0
            adjustment_made = False
            # Modify the cycle reset logic to retain a portion of the adjustments
            # 70% original value and 30% adjusted value
            cycle_durations[current_index] = int(1 * original_cycle_durations[current_index] + 0 * cycle_durations[current_index])


        # Set signal states for the next tick
        active_group = first_list[current_index]
        next_signals = []
        for signal in state.signals:
            if signal.name in active_group:
                next_signals.append(SignalDto(name=signal.name, state="green"))
                signal_state_durations[signal.name]["green"] += 1  # Log green state duration
                active_group_durations[signal.name] += 1  # Track how long it's been in the active group
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
