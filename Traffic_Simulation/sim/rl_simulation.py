from multiprocessing import Process, Queue
from time import sleep, time

from environment_gui import load_and_run_simulation

# Constants
STOPPED_SPEED_THRESHOLD = 0.5  # Speed below which a vehicle is considered stopped
MIN_GREEN_TIME = 10             # Minimum duration for a green phase (ticks)
MAX_GREEN_TIME = 60             # Maximum duration for a green phase (ticks)
QUEUE_DISTANCE_THRESHOLD = 50.0 # Distance to consider a vehicle as in queue
AVG_WAIT_TIME_WEIGHT = 0.5      # Weight for average waiting time in scoring
QUEUE_LENGTH_WEIGHT = 0.3       # Weight for queue length in scoring
FLOW_RATE_WEIGHT = 0.2          # Weight for vehicle flow rate in scoring

def calculate_amount_vehicle_in_leg(vehicles, leg_name):
    return len([vehicle for vehicle in vehicles if vehicle.leg == leg_name])

def calculate_stopped_vehicles(vehicles, leg_name):
    return len([vehicle for vehicle in vehicles if vehicle.leg == leg_name and vehicle.speed < STOPPED_SPEED_THRESHOLD])

def calculate_queue_length(vehicles, leg_name, distance_threshold=QUEUE_DISTANCE_THRESHOLD):
    return len([vehicle for vehicle in vehicles if vehicle.leg == leg_name and vehicle.distance_to_stop <= distance_threshold])

def calculate_average_wait_time(vehicles, leg_name):
    waiting_times = [vehicle.distance_to_stop / max(vehicle.speed, STOPPED_SPEED_THRESHOLD) for vehicle in vehicles if vehicle.leg == leg_name and vehicle.speed < STOPPED_SPEED_THRESHOLD]
    return sum(waiting_times) / len(waiting_times) if waiting_times else 0

def calculate_flow_rate(vehicles, leg_name, current_tick, previous_tick_data):
    # Calculate the number of vehicles that have entered the leg since the last tick
    previous_count = previous_tick_data.get(leg_name, 0)
    current_count = calculate_amount_vehicle_in_leg(vehicles, leg_name)
    flow_rate = current_count - previous_count
    previous_tick_data[leg_name] = current_count
    return max(flow_rate, 0)  # Ensure non-negative

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

    # Initialize variables
    current_phase = None          # Current allowed green signal combination name
    current_phase_duration = 0    # Duration of the current phase
    previous_tick_data = {}       # To store previous tick's vehicle counts for flow rate

    while True:
        state = output_queue.get()

        if state.is_terminated:
            p.join()
            break

        # Compute heuristics for each leg
        heuristics = {}
        for leg in state.legs:
            leg_name = leg.name
            stopped = calculate_stopped_vehicles(state.vehicles, leg_name)
            queue_length = calculate_queue_length(state.vehicles, leg_name)
            avg_wait_time = calculate_average_wait_time(state.vehicles, leg_name)
            flow_rate = calculate_flow_rate(state.vehicles, leg_name, state.simulation_ticks, previous_tick_data)
            heuristics[leg_name] = {
                'stopped': stopped,
                'queue_length': queue_length,
                'avg_wait_time': avg_wait_time,
                'flow_rate': flow_rate
            }

        # Map signals to legs
        signal_name_to_leg = {}
        for leg in state.legs:
            for signal_group in leg.signal_groups:
                signal_name_to_leg[signal_group] = leg.name

        # Compute score for each allowed green signal combination
        combination_scores = []
        for combo in state.allowed_green_signal_combinations:
            combo_name = combo.name
            signals_in_combo = combo.groups  # List of signal names
            legs_in_combo = set(signal_name_to_leg.get(signal, "") for signal in signals_in_combo)

            # Aggregate heuristics for legs in this combination
            total_stopped = sum(heuristics[leg]['stopped'] for leg in legs_in_combo)
            total_queue_length = sum(heuristics[leg]['queue_length'] for leg in legs_in_combo)
            avg_wait_time = sum(heuristics[leg]['avg_wait_time'] for leg in legs_in_combo)
            total_flow_rate = sum(heuristics[leg]['flow_rate'] for leg in legs_in_combo)

            # Weighted scoring
            score = (AVG_WAIT_TIME_WEIGHT * avg_wait_time +
                     QUEUE_LENGTH_WEIGHT * total_queue_length +
                     FLOW_RATE_WEIGHT * total_flow_rate)

            combination_scores.append((score, combo))

        # Sort combinations by score (highest first)
        combination_scores.sort(reverse=True, key=lambda x: x[0])

        # Get the best combination
        best_combo_score, best_combo = combination_scores[0]
        best_combo_name = best_combo.name
        best_combo_signals = best_combo.groups

        # Optional: Get second best for comparison or backup
        second_best_combo_score, second_best_combo = combination_scores[1] if len(combination_scores) > 1 else (None, None)

        # Decide whether to switch phases
        switch_phase = False
        if current_phase != best_combo_name:
            if current_phase_duration >= MIN_GREEN_TIME:
                switch_phase = True
        else:
            if current_phase_duration >= MAX_GREEN_TIME:
                switch_phase = True
            # Additionally, check if the best_combo score significantly exceeds others
            if second_best_combo and (best_combo_score - second_best_combo_score) > 10:  # Threshold can be tuned
                switch_phase = True

        if switch_phase:
            # Switch to new phase
            current_phase = best_combo_name
            current_phase_duration = 0
            # Update signals
            next_signals = {}
            for signal in state.signals:
                if signal.name in best_combo_signals:
                    next_signals[signal.name] = 'green'
                else:
                    next_signals[signal.name] = 'red'
            # Send the signal updates
            input_queue.put(next_signals)
        else:
            # Continue with current phase
            current_phase_duration += 1

        # For logging
        current_tick = state.simulation_ticks
        actions[current_tick] = {
            'phase': current_phase,
            'duration': current_phase_duration,
            'heuristics': heuristics
        }

    # End of simulation, return the score
    # Transform the score to the range [0, 1]
    if state.total_score == 0:
        state.total_score = 1e9

    return state.total_score


if __name__ == '__main__':
    run_game()
