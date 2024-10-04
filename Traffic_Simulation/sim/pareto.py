from multiprocessing import Process, Queue
from time import sleep, time
from environment import load_and_run_simulation
import uvicorn
from fastapi import FastAPI
import datetime
from loguru import logger
from pydantic import BaseModel
from dtos import TrafficSimulationPredictResponseDto, TrafficSimulationPredictRequestDto, SignalDto
import numpy as np
from typing import List, Dict
from pydantic import BaseModel
import numpy as np
from typing import List, Dict
import numpy as np
from pulp import LpMaximize, LpProblem, LpVariable, lpSum
import numpy as np
from pulp import LpMaximize, LpProblem, LpVariable, lpSum


##########################################

def extract_data_for_optimization(request: TrafficSimulationPredictRequestDto):
    """Extracts all vehicle speeds, distances, and assigns a small positive acceleration per leg.
    
    Args:
        request (TrafficSimulationPredictRequestDto): The current state of the simulation.
    
    Returns:
        Dictionary with leg names as keys and their corresponding lists of vehicle speeds, distances, and accelerations.
    """
    
    # Initialize a dictionary to hold vehicle data (speeds, distances, and accelerations) for each leg
    leg_data = {leg.name: {'speeds': [], 'distances': []} for leg in request.legs}
    
    # Accumulate data for each vehicle based on its leg
    for vehicle in request.vehicles:
        leg_name = vehicle.leg  # The leg the vehicle belongs to
        leg_data[leg_name]['speeds'].append(vehicle.speed)
        leg_data[leg_name]['distances'].append(vehicle.distance_to_stop)

    return leg_data


def estimate_acceleration_by_distance(speed, distance, current_phase, phase):  
    
    if speed == 0:
        return 0
    
    elif current_phase == phase:
        return 2.6
    elif current_phase != phase:
        return -4.6
    
   
def estimate_queue_length(leg_data, vth, dth):
    
    car_speeds = leg_data['speeds']
    car_distances = leg_data['distances']
    
    return sum(1 for v, d in zip(car_speeds, car_distances) if v < vth and d < dth)

def estimate_arrivals(leg_data, delta_t, current_phase, phase):
   
    speeds = leg_data['speeds']  # List of vehicle speeds in leg l
    distances = leg_data['distances']  # List of vehicle distances to stop line in leg l
    #accelerations = [estimate_acceleration_by_distance(d_i, v_i) for v_i, d_i in zip(speeds, distances)]
    accelerations = [estimate_acceleration_by_distance(speed, distance, current_phase, phase) for speed, distance in zip(speeds, distances)]
    #accelerations = [2.6] * len(speeds) if current_phase == phase else [-4.6]*len(speeds)
    
    arrivals = []
    for v_i, a_i in zip(speeds, accelerations):
        if a_i == 0:
            a_i = 0.01  # Avoid division by zero
        arrivals.append(np.sign(max(0, delta_t - v_i / a_i)))  # Sign function based on time to stop

    A_l = sum(arrivals)
    
    return A_l

def estimate_departures(leg_data, delta_t, saturation_flow_rate, current_phase_condition, yellow_time, current_phase, phase):
   
    # Condition c1: Use saturation flow rate
    if current_phase_condition == 'c1':
        return saturation_flow_rate * delta_t
    
    # Condition c2: The next phase to be green
    elif current_phase_condition == 'c2':
        speeds = leg_data['speeds']
        distances = leg_data['distances']
        #accelerations = [estimate_acceleration_by_distance(d_i, v_i) for v_i, d_i in zip(speeds, distances)]
        #accelerations = [2.6] * len(speeds) if current_phase == phase else [-4.6]*len(speeds)
        accelerations = [estimate_acceleration_by_distance(speed, distance, current_phase, phase) for speed, distance in zip(speeds, distances)]

        # Compute ΔL for each vehicle in the leg
        departures = []
        for v_i, d_i, a_i in zip(speeds, distances, accelerations):
            delta_L = v_i * (delta_t - yellow_time) + 0.5 * a_i * (delta_t - yellow_time)**2 - d_i
            # print("DELTA L: ", delta_L)
            departure = np.sign(max(0, -delta_L))  # Sign function based on ΔL
            departures.append(departure)
        
        # Sum the number of departures for all vehicles
        D_l = sum(departures)
        # print("------- GOT IN HEEEERE----- AND GOT: ", D_l)
        return D_l
    
    # No departing vehicles no matter what
    else: 
        return 0

def calculate_payoff(current_phase, leg_data, delta_t, saturation_flow_rate, yellow_time, phase):
    
    # Current phase is the phase which is currently green
    # Phase is the phase which we want to calculate the payoff for
   
    # Extract the speed and distances for the current phase
    current_phase_leg_data = list(leg_data.values())[phase]
    queue_length = estimate_queue_length(current_phase_leg_data, 1, 100)
    arrivals = estimate_arrivals(current_phase_leg_data, delta_t, current_phase, phase)
    
    # Print queue length and arrivals for the phase
    # print("SCORES FOR PHASE: ", phase)
    # print("QUEUE: ", queue_length)
    # print("ARRIVALS: ", arrivals)
    
    
    # We are in the current green phase
    if phase == current_phase:
        departures_stay = estimate_departures(current_phase_leg_data, delta_t, saturation_flow_rate, 'c1', yellow_time, current_phase, phase)
        departures_switch = estimate_departures(current_phase_leg_data, delta_t, saturation_flow_rate, 'switch', yellow_time, current_phase, phase)
    
    # We are the next up phase to be green
    elif phase == (current_phase+1) % 4:
        departures_stay = estimate_departures(current_phase_leg_data, delta_t, saturation_flow_rate, 'stay', yellow_time, current_phase, phase)
        departures_switch = estimate_departures(current_phase_leg_data, delta_t, saturation_flow_rate, 'c2', yellow_time, current_phase, phase)
    
    # We are in the other phases - Will get no departures no matter what
    else:
        departures_stay = 0
        departures_switch = 0
    
    
    # print("DEPARTURES (STAY): ", max(departures_stay, departures_switch))
    # print("DEPARTURES (SWITCH): ", min(departures_stay, departures_switch))
    
    # print("PAYOFF (STAY): ", queue_length + arrivals - departures_stay)
    # print("PAYOFF (SWITCH): ", queue_length + arrivals - departures_switch)
    # print()
    reward_stay = queue_length + arrivals - departures_stay
    reward_switch = queue_length + arrivals - departures_switch

    return reward_stay, reward_switch

def compute_pareto_solution(leg_data, current_phase, delta_t, saturation_flow_rate, yellow_time, min_green_time, max_green_time, green_durations, stay_will=0.6):
   
    # Check for minimum and maximum green time constraints
    if green_durations[current_phase] < min_green_time:
        return current_phase, False  # Stay in current phase
    
    if green_durations[current_phase] >= max_green_time:
        return (current_phase+1) % 4, True  # Force switch to next phase
    
    
    phases = [0, 1, 2, 3]  
    
    # Compute the payoffs for the current phase
    stay_current, switch_current = calculate_payoff(current_phase, 
                                                    leg_data, 
                                                    delta_t, 
                                                    saturation_flow_rate, 
                                                    yellow_time, 
                                                    current_phase)
    

    # Compute it for the next phase
    next_phase = (current_phase + 1) % len(phases)
    stay_next, switch_next = calculate_payoff(current_phase, 
                                              leg_data, 
                                              delta_t, 
                                              saturation_flow_rate, 
                                              yellow_time, 
                                              next_phase)
    
    # Compute it for the other phases
    other_1 = (current_phase + 2) % len(phases)
    other_2 = (current_phase + 3) % len(phases)
    
    stay_next_1, switch_next_1 = calculate_payoff(current_phase, 
                                                  leg_data, 
                                                  delta_t, 
                                                  saturation_flow_rate, 
                                                  yellow_time, 
                                                  other_1)
    
    stay_next_2, switch_next_2 = calculate_payoff(current_phase, 
                                                  leg_data, 
                                                  delta_t, 
                                                  saturation_flow_rate, 
                                                  yellow_time, 
                                                  other_2)
    
    
    # Combine next, 1 and 2 in a single coorporative player
    R_Pc_next = 0.6*stay_next + 0.3*stay_next_1 + 0.1*stay_next_2
    R_Pc_switch = 0.6*switch_next + 0.6*switch_next_1 + 0.6*switch_next_2 #Remember this
    
    stay_stay = 0.8*stay_current + 0.2*R_Pc_next
    switch_switch = 0.8*switch_current + 0.2*R_Pc_switch
    
    # print("FINAL PAYOFFS")
    # print("STAY_STAY: ", stay_stay)
    # print("SWITCH_SWITCH: ", switch_switch)
    
    
    if stay_will*stay_stay > (1-stay_will)*switch_switch:
        return current_phase, False
    else:
        return next_phase, True



def run_game():
    test_duration_seconds = 300
    random = True
    configuration_file = "models/1/glue_configuration.yaml"
    start_time = time()

    input_queue = Queue()
    output_queue = Queue()
    error_queue = Queue()
  

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
    
    num_phases = 4  # Number of phases in the signal cycle
    current_phase = 0  # Start with phase P1
    delta_t = 6                 #BO 5-24 (2 intervals)
    saturation_flow_rate = 1.1  #BO 0.6-2 (0.2 intervals)
    yellow_time = 3             #BO 2,3,4 (1 interval)
    min_green_time = 6          #BO 6-15 (3 intervals)
    max_green_time = 30         #BO 30-50 (5 intervals)
    stay_will = 0.6             #BO 0.5-0.9 (0.1 intervals)

    # Define the signal groups for each phase
    phase_signals = {
        0: ['A1', 'A1LeftTurn'],  # Phase 1
        1: ['A2', 'A2LeftTurn'],  # Phase 3
        2: ['B1', 'B1LeftTurn'],  # Phase 2
        3: ['B2', 'B2LeftTurn']   # Phase 4
    }

    green_durations = [0] * num_phases  # Track how long each phase has been green

    while True:
        state = output_queue.get()

        if state.is_terminated:
            break

        # Extract vehicle speeds, distances, and accelerations per leg
        leg_data = extract_data_for_optimization(state)

        # Call the Pareto solution to decide whether to stay or switch phase
        next_phase, should_switch = compute_pareto_solution(leg_data, 
                                                            current_phase, 
                                                            delta_t, 
                                                            saturation_flow_rate, 
                                                            yellow_time, 
                                                            min_green_time, 
                                                            max_green_time, 
                                                            green_durations,
                                                            stay_will)

        if should_switch:
            current_phase = next_phase
            green_durations = [0] * num_phases  # Reset all green durations when switching
        else:
            green_durations[current_phase] += 1  # Increment time for the current green phase

        # print(green_durations)


        # Prepare the response with the correct signals for the new phase
        green_signal_group = phase_signals[current_phase]  # Get the signal group for the current phase
        all_signals = state.signal_groups  # List of all signal groups
        
        # Set the signals to green for the current phase and red for others
        response_signals = {}
        for signal_group in all_signals:
            signal_state = "green" if signal_group in green_signal_group else "red"
            response_signals[signal_group] = signal_state

 
        # Send the updated signal data to SUMO
        input_queue.put(response_signals)
        
   
    # print("WE GOT HERE")
    print(state.total_score)
    return
    

if __name__ == '__main__':
    run_game()