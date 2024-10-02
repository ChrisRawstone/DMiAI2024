from multiprocessing import Process, Queue
from time import sleep, time
from typing import List, Dict

from environment_gui import load_and_run_simulation

# Define threshold for stopped vehicles
STOPPED_SPEED_THRESHOLD = 0.5  # Speed below which a vehicle is considered stopped
QUEUE_DISTANCE_THRESHOLD = 50.0  # Distance within which to consider vehicles for queue length

# Heuristic functions
def calculate_amount_vehicle_in_leg(vehicles: List['VehicleDto'], leg_name: str) -> int:
    return len([vehicle for vehicle in vehicles if vehicle.leg == leg_name])

def calculate_stopped_vehicles(vehicles: List['VehicleDto'], leg_name: str) -> int:
    return len([vehicle for vehicle in vehicles if vehicle.leg == leg_name and vehicle.speed < STOPPED_SPEED_THRESHOLD])

def calculate_queue_length(vehicles: List['VehicleDto'], leg_name: str, distance_threshold: float = QUEUE_DISTANCE_THRESHOLD) -> int:
    return len([vehicle for vehicle in vehicles if vehicle.leg == leg_name and vehicle.distance_to_stop <= distance_threshold])

def run_game():
    test_duration_seconds = 600
    random = True
    configuration_file = "models/1/glue_configuration.yaml"
    start_time = time()

    input_queue = Queue()
    output_queue = Queue()
    error_queue = Queue()
    errors = []

    p = Process(target=load_and_run_simulation, args=(
        configuration_file,
        start_time,
        test_duration_seconds,
        random,
        input_queue,
        output_queue,
        error_queue
    ))
    
    p.start()

    # Wait for the simulation to start
    sleep(0.2)

    # For logging
    actions = {}
    leg_waiting_ticks = {}  # To track consecutive ticks with stopped vehicles per leg

    while True:
        state = output_queue.get()

        if state.is_terminated:
            p.join()
            break

        current_tick = state.simulation_ticks

        # print(f'--- Tick: {current_tick} ---')
        # print(f'Vehicles: {state.vehicles}')
        # print(f'Signals: {state.signals}')

        # Initialize metrics dictionary
        metrics: Dict[str, Dict[str, float]] = {}

        for leg in state.legs:
            leg_name = leg.name
            vehicles_in_leg = [v for v in state.vehicles if v.leg == leg_name]
            stopped_count = calculate_stopped_vehicles(state.vehicles, leg_name)
            queue_length = calculate_queue_length(state.vehicles, leg_name)

            num_vehicles = calculate_amount_vehicle_in_leg(state.vehicles, leg_name)

            # Update waiting ticks
            if leg_name not in leg_waiting_ticks:
                leg_waiting_ticks[leg_name] = 0

            if stopped_count > 0:
                leg_waiting_ticks[leg_name] += 1
            else:
                leg_waiting_ticks[leg_name] = 0  # Reset if no vehicles are stopped

            # Calculate heuristics
            total_wait = leg_waiting_ticks[leg_name]
            avg_wait = total_wait / stopped_count if stopped_count > 0 else 0
            max_wait = total_wait  # Simplification since we can't track individual vehicles

            # Store metrics
            metrics[leg_name] = {
                'num_vehicles': num_vehicles,
                'num_stopped': stopped_count,
                'total_waiting_time': total_wait,
                'average_waiting_time': avg_wait,
                'max_waiting_time': max_wait,
                'queue_length': queue_length
            }

            # Print metrics for debugging
            print(f'Leg {leg_name}: {metrics[leg_name]}')

        # Example: Use heuristics to decide signal states
        prediction = {"signals": []}

        for leg_name, data in metrics.items():
            # Simple heuristic: if total_waiting_time exceeds threshold, set signal to green
            TOTAL_WAIT_THRESHOLD = 10  # Define an appropriate threshold based on simulation ticks
            if data['total_waiting_time'] > TOTAL_WAIT_THRESHOLD:
                prediction["signals"].append({"name": leg_name, "state": "green"})
            else:
                prediction["signals"].append({"name": leg_name, "state": "red"})

        # Ensure only allowed signal combinations are activated
        # Implement logic based on allowed_green_signal_combinations
        # For simplicity, assume no conflicts in this example

        # Update the desired phase of the traffic lights
        next_signals = {}
        for signal in prediction['signals']:
            actions[current_tick] = (signal['name'], signal['state'])
            next_signals[signal['name']] = signal['state']

        input_queue.put(next_signals)

        # Handle any errors from error_queue
        try:
            while not error_queue.empty():
                signal_logic_errors = error_queue.get_nowait()
                if signal_logic_errors:
                    errors.append(signal_logic_errors)
        except:
            pass

    # End of simulation, return the score

    # Transform the score to the range [0, 1]
    if state.total_score == 0:
        state.total_score = 1e9

    inverted_score = 1. / state.total_score

    return inverted_score

if __name__ == '__main__':
    run_game()
