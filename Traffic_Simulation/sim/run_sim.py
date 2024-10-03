from multiprocessing import Process, Queue
from time import sleep, time
from typing import List, Dict

from environment_gui import load_and_run_simulation
from typing import Dict

# Define threshold for stopped vehicles
STOPPED_SPEED_THRESHOLD = 0.5  # Speed below which a vehicle is considered stopped
QUEUE_DISTANCE_THRESHOLD = 50.0  # Distance within which to consider vehicles for queue length

global time_since_signals_has_been_green, wait_before_start_ticks

# Heuristic functions
def calculate_amount_vehicle_in_leg(vehicles: List['VehicleDto'], leg_name: str) -> int:
    return len([vehicle for vehicle in vehicles if vehicle.leg == leg_name])

def calculate_stopped_vehicles(vehicles: List['VehicleDto'], leg_name: str) -> int:
    return len([vehicle for vehicle in vehicles if vehicle.leg == leg_name and vehicle.speed < STOPPED_SPEED_THRESHOLD])

def calculate_queue_length(vehicles: List['VehicleDto'], leg_name: str, distance_threshold: float = QUEUE_DISTANCE_THRESHOLD) -> int:
    return len([vehicle for vehicle in vehicles if vehicle.leg == leg_name and vehicle.distance_to_stop <= distance_threshold])

def get_unique_signals(state) -> List[str]:
    unique_signals = set()
    for leg in state.legs:
        for lane in leg.lanes:
            unique_signals.add(lane)
    return list(unique_signals)



def calculate_leg_metrics(state, leg_waiting_ticks: Dict[str, int]) -> Dict[str, Dict[str, float]]:
    """
    Calculates various traffic metrics for each leg in the given state.

    Args:
        state: The current state of the traffic simulation.
        leg_waiting_ticks (Dict[str, int]): Dictionary to keep track of waiting ticks for each leg.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary containing the calculated metrics for each leg.
    """
    metrics: Dict[str, Dict[str, float]] = {}

    for leg in state.legs:
        leg_name = leg.name
        stopped_count = calculate_stopped_vehicles(state.vehicles, leg_name)
        queue_length = calculate_queue_length(state.vehicles, leg_name)
        num_vehicles = calculate_amount_vehicle_in_leg(state.vehicles, leg_name)

        # Update waiting ticks
        if leg_name not in leg_waiting_ticks:
            leg_waiting_ticks[leg_name] = 0

        if stopped_count > 0:
            leg_waiting_ticks[leg_name] += stopped_count
        else:
            leg_waiting_ticks[leg_name] = 0  # Reset if no vehicles are stopped

        total_wait = leg_waiting_ticks[leg_name]
        avg_wait = total_wait / stopped_count if stopped_count > 0 else 0

        metrics[leg_name] = {
            'num_vehicles': num_vehicles,
            'num_stopped': stopped_count,
            'total_waiting_time': total_wait,
            'average_waiting_time': avg_wait,
            'queue_length': queue_length
        }

    return metrics, leg_waiting_ticks


wait_before_start_ticks = 20

time_since_signals_has_been_green = 1000




def run_game():
    global time_since_signals_has_been_green, wait_before_start_ticks

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
    sleep(0.2)  # Wait for the simulation to start

    actions = {}
    leg_waiting_ticks = {}

    

    while True:
        state = output_queue.get()
        unique_signals = get_unique_signals(state)

        if state.is_terminated:
            p.join()
            break

        current_tick = state.simulation_ticks

        print("############################")
        print("Current tick: ", current_tick)
        print("############################")

        metrics, leg_waiting_ticks = calculate_leg_metrics(state, leg_waiting_ticks)


        if time_since_signals_has_been_green > 1:
            # Sort legs by total waiting time (descending order)
            time_since_signals_has_been_green = 0
            sorted_legs = sorted(metrics.items(), key=lambda x: x[1]['total_waiting_time'], reverse=True)

            # find the leg with most waiting time
            signal_with_most_waiting_time = sorted_legs[0][0]

            # Find allowed combinations that include the top two legs
            chosen_combination = None
            for combination in state.allowed_green_signal_combinations:
                if signal_with_most_waiting_time==combination.name:
                        signal_with_2nd_most_waiting_time = max(combination.groups, key=lambda leg: metrics.get(leg, {}).get('total_waiting_time', 0))
                        chosen_combination = (signal_with_most_waiting_time, signal_with_2nd_most_waiting_time)
                        
            print("turning ", chosen_combination, "green")
            # if state.simulation_ticks > wait_before_start_ticks:

        # If a valid combination is found, activate it
        prediction = {"signals": []}
        if chosen_combination:
            for signal in chosen_combination:
                prediction["signals"].append({"name": signal, "state": "green"})

            



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
        time_since_signals_has_been_green += 1


    if state.total_score == 0:
        state.total_score = 1e9

    inverted_score = 1. / state.total_score
    return inverted_score


if __name__ == '__main__':
    run_game()
