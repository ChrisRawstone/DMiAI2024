from multiprocessing import Process, Queue
from time import sleep, time
from typing import List, Dict

from environment_gui import load_and_run_simulation
from typing import Dict
from collections import defaultdict

#defining global variables
global time_since_signals_has_been_green, wait_before_start_ticks, metrics, previous_green_signals, time_turned_green 
global default_time_turned_green

# Define threshold for stopped vehicles
STOPPED_SPEED_THRESHOLD = 0.5  # Speed below which a vehicle is considered stopped
QUEUE_DISTANCE_THRESHOLD = 50.0  # Distance within which to consider vehicles for queue length

previous_green_signals = []

time_turned_green = 1
default_time_turned_green = 6

delay = 0
wait_before_start_ticks = 15
time_since_signals_has_been_green = 1000

metrics = defaultdict(dict)



# Helper functions
def calculate_amount_vehicle_in_leg(vehicles: List['VehicleDto'], leg_name: str) -> int:
    return len([vehicle for vehicle in vehicles if vehicle.leg == leg_name])


def get_green_signals(signals) -> List[str]:   
    return [signal.name for signal in signals if signal.state == 'green']

def get_vehicle_in_specific_leg(vehicles: List['VehicleDto'], leg_name: str) -> int:
    return [vehicle for vehicle in vehicles if vehicle.leg == leg_name]

def calculate_stopped_vehicles(vehicles: List['VehicleDto'], leg_name: str) -> int:
    return len([vehicle for vehicle in vehicles if vehicle.leg == leg_name and vehicle.speed < STOPPED_SPEED_THRESHOLD])

# def calculate_queue_length(vehicles: List['VehicleDto'], leg_name: str, distance_threshold: float = QUEUE_DISTANCE_THRESHOLD) -> int:
#     return len([vehicle for vehicle in vehicles if vehicle.leg == leg_name and vehicle.distance_to_stop <= distance_threshold])

def get_signal_groups(name, legs):
    for leg in legs:
        if leg.name == name:
            return leg.signal_groups
    AssertionError




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
    signal_with_most_waiting_time = sorted_legs[0][0] #A1, B2 
    
    # If signal is in corresponding_sidelanes, get the corresponding sideline
    # next_signal = corresponding_sidelanes[signal_with_most_waiting_time]
    next_signals = get_signal_groups(signal_with_most_waiting_time, state.legs)

    # get all signals in next_signals that is not signal_with_most_waiting_time
    next_signals_cleaned = [signal for signal in next_signals if signal != signal_with_most_waiting_time]
    
    # unpack the next_signals
    try:
        return signal_with_most_waiting_time, next_signals_cleaned[0], next_signals_cleaned[1]
    except:
        return signal_with_most_waiting_time, next_signals_cleaned[0]







def initialize_metrics(state):
    global metrics
    for signal in state.legs:
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

        
        metrics[leg_name] = {
            'num_stopped': (stopped_count_majorlane)}



def turn_signal_green(prediction, signal):
    prediction["signals"].append({"name": signal, "state": "green"})
    return prediction





def run_game():
    global time_since_signals_has_been_green, wait_before_start_ticks, metrics, previous_green_signals, time_turned_green, default_time_turned_green

    test_duration_seconds = 300
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
        

        if (time_since_signals_has_been_green-delay) > default_time_turned_green+time_turned_green:
            time_turned_green = 1
            
            update_major_metrics(state)
            
            time_since_signals_has_been_green = 0
        
            chosen_combination = find_two_signals_with_most_waiting_time(state) # A1, A2, B1, B2

            
            time_turned_green += time_turned_green*round(len(get_vehicle_in_specific_leg(state.vehicles, chosen_combination[0]))*0.2)
                


            

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

    print(state.total_score)
    return state.total_score


if __name__ == '__main__':
    run_game()
