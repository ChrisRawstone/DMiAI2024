import uvicorn
from fastapi import FastAPI
import datetime
import time
from loguru import logger
from pydantic import BaseModel
from sim.dtos import TrafficSimulationPredictResponseDto, TrafficSimulationPredictRequestDto, SignalDto




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





def get_green_signals(signals):   
    return [signal.name for signal in signals if signal.state == 'green']

def get_vehicle_in_specific_leg(vehicles, leg_name):
    return [vehicle for vehicle in vehicles if vehicle.leg == leg_name]

def calculate_stopped_vehicles(vehicles, leg_name):
    return len([vehicle for vehicle in vehicles if vehicle.leg == leg_name and vehicle.speed < STOPPED_SPEED_THRESHOLD])


# def calculate_queue_length(vehicles: List['VehicleDto'], leg_name: str, distance_threshold: float = QUEUE_DISTANCE_THRESHOLD) -> int:
#     return len([vehicle for vehicle in vehicles if vehicle.leg == leg_name and vehicle.distance_to_stop <= distance_threshold])

def get_signal_groups(name, legs):
    for leg in legs:
        if leg.name == name:
            return leg.signal_groups
    AssertionError




def get_unique_signals(legs, signal_groups):
    unique_signals = set()
    for leg in legs:
        for signal in signal_groups:
            unique_signals.add(signal)
    return list(unique_signals)

def find_two_signals_with_most_waiting_time(legs):
    global metrics
    
    heuristic = "num_stopped"

    sorted_legs = sorted(metrics.items(), key=lambda x: x[1][heuristic], reverse=True)
    signal_with_most_waiting_time = sorted_legs[0][0] #A1, B2 
    
    # If signal is in corresponding_sidelanes, get the corresponding sideline
    # next_signal = corresponding_sidelanes[signal_with_most_waiting_time]
    next_signals = get_signal_groups(signal_with_most_waiting_time, legs)

    # get all signals in next_signals that is not signal_with_most_waiting_time
    next_signals_cleaned = [signal for signal in next_signals if signal != signal_with_most_waiting_time]
    
    # unpack the next_signals
    try:
        return signal_with_most_waiting_time, next_signals_cleaned[0], next_signals_cleaned[1]
    except:
        return signal_with_most_waiting_time, next_signals_cleaned[0]







def initialize_metrics(legs):
    global metrics
    for signal in legs:
        metrics[signal.name] = {
            'num_stopped': 0,
        }

def update_major_metrics(legs, vehicles):
    """
    Calculates various traffic metrics for each leg in the given state.

    Args:
        state: The current state of the traffic simulation.
        leg_waiting_ticks (Dict[str, int]): Dictionary to keep track of waiting ticks for each leg.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary containing the calculated metrics for each leg.
    """
    global metrics

    for leg in legs: # A1, A2, B1, B2
        leg_name = leg.name
        stopped_count_majorlane = calculate_stopped_vehicles(vehicles, leg_name)

        
        metrics[leg_name] = {
            'num_stopped': (stopped_count_majorlane)}



def turn_signal_green(prediction, signal):
    prediction["signals"].append({"name": signal, "state": "green"})
    return prediction



HOST = "0.0.0.0"
PORT = 8080

app = FastAPI()
start_time = time.time()

@app.get('/api')
def hello():
    return {
        "service": "traffic-simulation-usecase",
        "uptime": '{}'.format(datetime.timedelta(seconds=time.time() - start_time))
    }

@app.get('/')
def index():
    return "Your endpoint is running!"

@app.post('/predict', response_model=TrafficSimulationPredictResponseDto)
def predict_endpoint(request: TrafficSimulationPredictRequestDto):
    global time_since_signals_has_been_green, wait_before_start_ticks, metrics, previous_green_signals, time_turned_green, default_time_turned_green


    # Decode request
    data = request
    vehicles = data.vehicles
    total_score = data.total_score
    simulation_ticks = data.simulation_ticks
    signals = data.signals
    signal_groups = data.signal_groups
    legs = data.legs
    allowed_green_signal_combinations = data.allowed_green_signal_combinations
    is_terminated = data.is_terminated
    
    unique_signals = get_unique_signals(legs, signal_groups)

    if simulation_ticks <= 10:
        initialize_metrics(legs)

    print("############################")
    print("Current tick: ", simulation_ticks)
    print("############################")

    chosen_combination = None
    

    if (time_since_signals_has_been_green-delay) > default_time_turned_green+time_turned_green:
        time_turned_green = 1
        
        update_major_metrics(legs, vehicles)
        
        time_since_signals_has_been_green = 0
    
        chosen_combination = find_two_signals_with_most_waiting_time(legs) # A1, A2, B1, B2

        
        time_turned_green += time_turned_green*round(len(get_vehicle_in_specific_leg(vehicles, chosen_combination[0]))*0.2)
            


        

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
    actions = {}
    next_signals = {}
    for signal in prediction['signals']:
        actions[simulation_ticks] = (signal['name'], signal['state'])
        next_signals[signal['name']] = signal['state']








    # Select a signal group to go green
    # green_signal_group = signal_groups[0]

    api_signals = []
    for pair in list(next_signals.items()):
        api_signals.append(SignalDto(name=pair[0], state=pair[1]))


    print(total_score)

    # Return the encoded image to the validation/evalution service
    response = TrafficSimulationPredictResponseDto(
        signals=api_signals
    )

    return response

if __name__ == '__main__':

    uvicorn.run(
        'api:app',
        host=HOST,
        port=PORT
    )