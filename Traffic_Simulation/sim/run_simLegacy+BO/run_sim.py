# Import necessary libraries
from multiprocessing import Process, Queue
from environment import load_and_run_simulation
from collections import defaultdict, deque
from dtos import SignalDto
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

# Define the signal combinations
first_list = [
    ['A1', 'A1LeftTurn'],
    ['A2', 'A2LeftTurn'],
    ['B1', 'B1LeftTurn'],
    ['B2', 'B2LeftTurn']
]

# Define the main simulation function
def run_game(original_cycle_durations, ticks_reduction):
    # Initialize queues inside the function
    input_queue = Queue()
    output_queue = Queue()
    error_queue = Queue()

    # Initialize variables
    cycle_durations = original_cycle_durations.copy()
    ticks_per_group = 0  # Tracks how many ticks each group has been active
    current_index = 0  # Tracks which signal group is currently active
    duration_reduced = False  # Boolean flag to ensure reduction happens only once per cycle

    # Track vehicle counts for each leg for the last 5 ticks
    vehicle_counts_history = defaultdict(lambda: deque(maxlen=5))

    # Logging variables for signal state durations
    signal_state_durations = defaultdict(lambda: defaultdict(int))

    # Main simulation code
    test_duration_seconds = 300  # Adjust as needed for faster testing
    random = False
    configuration_file = "models/1/glue_configuration.yaml"

    # Start simulation process
    p = Process(target=load_and_run_simulation, args=(
        configuration_file, None, test_duration_seconds, random, input_queue, output_queue, error_queue))
    p.start()

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
        leg_vehicle_count = sum(1 for vehicle in state.vehicles if vehicle.leg == current_leg)

        # Store the current vehicle count in the history for the active leg
        vehicle_counts_history[current_leg].append(leg_vehicle_count)

        # Check if we have 5 ticks of history for the active leg
        if len(vehicle_counts_history[current_leg]) == 5:
            # Calculate the change in vehicle count over the last 5 ticks
            change_in_vehicle_count = vehicle_counts_history[current_leg][-1] - vehicle_counts_history[current_leg][0]
            # Temporarily adjust the current cycle duration if the condition is met
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

    return state.total_score

# Function to log parameters and score to a file
def log_to_file(parameters, score, filename="simulation_log.txt"):
    with open(filename, "a") as f:
        f.write(f"Parameters: {parameters}\n")
        f.write(f"Total score: {score}\n")
        f.write("="*40 + "\n")

# Define the parameter space for optimization
space = [
    Integer(10, 50, name='d1'),
    Integer(10, 50, name='d2'),
    Integer(10, 50, name='d3'),
    Integer(10, 50, name='d4'),
    Integer(1, 20, name='ticks_reduction'),
]

# Define the objective function for optimization
@use_named_args(space)
def objective(**params):
    d1 = params['d1']
    d2 = params['d2']
    d3 = params['d3']
    d4 = params['d4']
    ticks_reduction = params['ticks_reduction']
    original_cycle_durations = [d1, d2, d3, d4]
    
    # Log the used parameters
    print(f"Running simulation with parameters: cycle_durations={original_cycle_durations}, ticks_reduction={ticks_reduction}")

    # Run the simulation and get the score
    total_score = run_game(original_cycle_durations, ticks_reduction)
    
    # Log to file
    log_to_file(parameters={
        'cycle_durations': original_cycle_durations,
        'ticks_reduction': ticks_reduction
    }, score=total_score)

    # Print the cost (negative total score since skopt minimizes)
    print(f"Simulation result: total_score={total_score}, cost={-total_score}")

    # Return negative total score for minimization
    return total_score  # Negate because skopt minimizes by default

if __name__ == '__main__':
    # Run Bayesian optimization
    res = gp_minimize(objective, space, n_calls=150, random_state=0)

    # Print the best result
    print("Best total score=%.4f" % -res.fun)
    print("Best parameters:")
    print("d1=%d, d2=%d, d3=%d, d4=%d, ticks_reduction=%d" % (
        res.x[0], res.x[1], res.x[2], res.x[3], res.x[4]))
