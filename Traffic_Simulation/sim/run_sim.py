from multiprocessing import Process, Queue
from time import sleep, time
from typing import List, Dict

from environment_gui import load_and_run_simulation
from typing import Dict
from collections import defaultdict

#defining global variables
global time_since_signals_has_been_green, wait_before_start_ticks, metrics, previous_green_signals

# Define threshold for stopped vehicles
STOPPED_SPEED_THRESHOLD = 0.5  # Speed below which a vehicle is considered stopped
QUEUE_DISTANCE_THRESHOLD = 50.0  # Distance within which to consider vehicles for queue length

previous_green_signals = []

delay = 0
wait_before_start_ticks = 15
time_since_signals_has_been_green = 1000

metrics = defaultdict(dict)

pairs_of_signals = {
    "A1" : "A2",
    "A2" : "A1",  
    "A1LeftTurn" : "A2LeftTurn",
    "A2LeftTurn" : "A1LeftTurn",  
    "B1" : "B2",
    "B2" : "B1",  
    "B1LeftTurn" : "B2LeftTurn",
    "B2LeftTurn" : "B1LeftTurn"  
}


corresponding_sidelanes = {
    "A1" : "A1LeftTurn",
    "A2" : "A2LeftTurn",
    "B1" : "B1LeftTurn",
    "B2" : "B2LeftTurn"
}
inverse_sidelanes = {v: k for k, v in corresponding_sidelanes.items()}


# Helper functions
def calculate_amount_vehicle_in_leg(vehicles: List['VehicleDto'], leg_name: str) -> int:
    return len([vehicle for vehicle in vehicles if vehicle.leg == leg_name])


def get_green_signals(signals) -> List[str]:   
    return [signal.name for signal in signals if signal.state == 'green']

def get_vehicle_in_specific_leg(vehicles: List['VehicleDto'], leg_name: str) -> int:
    return [vehicle for vehicle in vehicles if vehicle.leg == leg_name]

def calculate_stopped_vehicles(vehicles: List['VehicleDto'], leg_name: str) -> int:
    return len([vehicle for vehicle in vehicles if vehicle.leg == leg_name and vehicle.speed < STOPPED_SPEED_THRESHOLD])

def calculate_queue_length(vehicles: List['VehicleDto'], leg_name: str, distance_threshold: float = QUEUE_DISTANCE_THRESHOLD) -> int:
    return len([vehicle for vehicle in vehicles if vehicle.leg == leg_name and vehicle.distance_to_stop <= distance_threshold])

def get_unique_signals(state) -> List[str]:
    unique_signals = set()
    for leg in state.legs:
        for signal in leg.signal_groups:
            unique_signals.add(signal)
    return list(unique_signals)

def find_two_signals_with_most_waiting_time(state):
    global metrics
    
    heuristic = "num_stopped"

    sorted_legs = sorted(metrics.items(), key=lambda x: x[1][heuristic], reverse=True)
    signal_with_most_waiting_time = sorted_legs[0][0]
    
    chosen_combination = None
    for combination in state.allowed_green_signal_combinations:
        if signal_with_most_waiting_time==combination.name:
                signal_with_2nd_most_waiting_time = max([leg for leg in combination.groups if leg in metrics], key=lambda leg: metrics[leg][heuristic])
                chosen_combination = (signal_with_most_waiting_time, signal_with_2nd_most_waiting_time)

    # if signal_with_most_waiting_time in corresponding_sidelanes:
    #     # If signal is in corresponding_sidelanes, get the corresponding sideline
    #     next_signal = corresponding_sidelanes[signal_with_most_waiting_time]
    # else:
    #     next_signal = inverse_sidelanes[signal_with_most_waiting_time]

    # return (signal_with_most_waiting_time, next_signal)
    return chosen_combination

def initialize_metrics(state):
    global metrics
    for signal in state.signals:
        metrics[signal.name] = {
            'num_stopped': 0,
        }

def update_major_metrics(state):
    """
    Calculates various traffic metrics for each leg in the given state.

    Args:
        state: The current state of the traffic simulation.
        leg_waiting_ticks (Dict[str, int]): Dictionary to keep track of waiting ticks for each leg.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary containing the calculated metrics for each leg.
    """
    global metrics

    for leg in state.legs: # A1, A2, B1, B2
        leg_name = leg.name
        stopped_count_majorlane = calculate_stopped_vehicles(state.vehicles, leg_name)

        sidelane_signal = corresponding_sidelanes[leg_name]

        stopped_count_sidelane = metrics[sidelane_signal]["num_stopped"]
        
        metrics[leg_name] = {
            'num_stopped': (stopped_count_majorlane-stopped_count_sidelane)}

def get_health_sidelane(state, signal):

    vehicles_current_leg = get_vehicle_in_specific_leg(state.vehicles, signal)

    # Finding the vehicle with the highest distance and speed equal to 0
    try:
        max_distance_vehicle = max([v for v in vehicles_current_leg if v.speed == 0.0], key=lambda x: x.distance_to_stop).distance_to_stop
    except:
        max_distance_vehicle = 0

    # Calculating the proportion of cars in speed (speed > 0) vs all cars and cars not in speed (speed = 0) vs all cars
    total_cars = len(vehicles_current_leg)
    cars_in_speed = len([v for v in vehicles_current_leg if v.speed > 0])
    cars_not_in_speed = len([v for v in vehicles_current_leg if v.speed == 0])

    proportion_in_speed = cars_in_speed / total_cars
    proportion_not_in_speed = cars_not_in_speed / total_cars
    
    sidelane_queue_length = max_distance_vehicle
    num_cars_in_sidelane = cars_not_in_speed

    return sidelane_queue_length, num_cars_in_sidelane, proportion_in_speed, proportion_not_in_speed, 

def turn_signal_green(prediction, signal):
    prediction["signals"].append({"name": signal, "state": "green"})
    return prediction

def update_minor_metrics(state, current_green_signals):
    for green_signal in current_green_signals: # remember this only works with major green signals, fix this
        
        if len(get_vehicle_in_specific_leg(state.vehicles, green_signal)) > 0:
            sidelane_queue_length, num_cars_in_sidelane, proportion_in_speed, proportion_not_in_speed = get_health_sidelane(state, green_signal)
            num_cars_in_major_lane = metrics[green_signal]["num_stopped"]


            sidelane_signal = corresponding_sidelanes[green_signal]

            metrics[sidelane_signal]["num_stopped"] = num_cars_in_sidelane
            metrics[green_signal]["num_stopped"] = num_cars_in_major_lane-num_cars_in_sidelane


def has_green_signals_changed(state):
    global previous_green_signals
    if get_green_signals(state.signals) != previous_green_signals:
        previous_green_signals = get_green_signals(state.signals)
        return True
    else:
        return False


def run_game():
    global time_since_signals_has_been_green, wait_before_start_ticks, metrics, previous_green_signals

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
        error_queue))
    
    p.start()
    sleep(0.2)  # Wait for the simulation to start

    actions = {}

    while True:
        state = output_queue.get()

        unique_signals = get_unique_signals(state)

        if state.is_terminated:
            p.join()
            break

        current_tick = state.simulation_ticks

        if current_tick <= 10:
            initialize_metrics(state)

        print("############################")
        print("Current tick: ", current_tick)
        print("############################")

        chosen_combination = None

        if current_tick > wait_before_start_ticks:
            if (time_since_signals_has_been_green-delay) > 6:
                
                update_major_metrics(state)
                
                time_since_signals_has_been_green = 0
            
                chosen_combination = find_two_signals_with_most_waiting_time(state) # A1, A2, B1, B2
            

        # If a valid combination is found, activate it
        prediction = {"signals": []}
        if chosen_combination:
            print("turning ", chosen_combination, "green")
            for signal in chosen_combination:
                prediction = turn_signal_green(prediction, signal)
                
            # turn all signals red for 
            for signal in unique_signals:
                if signal not in chosen_combination:
                    prediction["signals"].append({"name": signal, "state": "red"})

        print(metrics)


        current_green_signals = get_green_signals(state.signals)
        
        if has_green_signals_changed(state):
            update_minor_metrics(state, current_green_signals)







            


        
        





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
