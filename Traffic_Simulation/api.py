import uvicorn
from fastapi import FastAPI
import datetime
from loguru import logger
from pydantic import BaseModel
from sim.dtos import TrafficSimulationPredictResponseDto, TrafficSimulationPredictRequestDto, SignalDto
from collections import defaultdict, deque
from multiprocessing import Queue
import numpy as np



#### FOR MAP 1 ####
def extract_data_for_optimization1(request: TrafficSimulationPredictRequestDto):
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

def estimate_acceleration_by_distance1(speed, distance, current_phase, phase):  
    
    if speed == 0:
        return 0
    
    elif current_phase == phase:
        return 2.6
    elif current_phase != phase:
        return -4.6
     
def estimate_queue_length1(leg_data, vth, dth):
    
    car_speeds = leg_data['speeds']
    car_distances = leg_data['distances']
    
    return sum(1 for v, d in zip(car_speeds, car_distances) if v < vth and d < dth)

def estimate_arrivals1(leg_data, delta_t, current_phase, phase):
   
    speeds = leg_data['speeds']  # List of vehicle speeds in leg l
    distances = leg_data['distances']  # List of vehicle distances to stop line in leg l
    #accelerations = [estimate_acceleration_by_distance(d_i, v_i) for v_i, d_i in zip(speeds, distances)]
    accelerations = [estimate_acceleration_by_distance1(speed, distance, current_phase, phase) for speed, distance in zip(speeds, distances)]
    #accelerations = [2.6] * len(speeds) if current_phase == phase else [-4.6]*len(speeds)
    
    arrivals = []
    for v_i, a_i in zip(speeds, accelerations):
        if a_i == 0:
            a_i = 0.01  # Avoid division by zero
        arrivals.append(np.sign(max(0, delta_t - v_i / a_i)))  # Sign function based on time to stop

    A_l = sum(arrivals)
    
    return A_l

def estimate_departures1(leg_data, delta_t, saturation_flow_rate, current_phase_condition, yellow_time, current_phase, phase):
   
    # Condition c1: Use saturation flow rate
    if current_phase_condition == 'c1':
        return saturation_flow_rate * delta_t
    
    # Condition c2: The next phase to be green
    elif current_phase_condition == 'c2':
        speeds = leg_data['speeds']
        distances = leg_data['distances']
        #accelerations = [estimate_acceleration_by_distance(d_i, v_i) for v_i, d_i in zip(speeds, distances)]
        #accelerations = [2.6] * len(speeds) if current_phase == phase else [-4.6]*len(speeds)
        accelerations = [estimate_acceleration_by_distance1(speed, distance, current_phase, phase) for speed, distance in zip(speeds, distances)]

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

def calculate_payoff1(current_phase, leg_data, delta_t, saturation_flow_rate, yellow_time, phase):
    
    # Current phase is the phase which is currently green
    # Phase is the phase which we want to calculate the payoff for
   
    # Extract the speed and distances for the current phase
    current_phase_leg_data = list(leg_data.values())[phase]
    queue_length = estimate_queue_length1(current_phase_leg_data, 1, 100)
    arrivals = estimate_arrivals1(current_phase_leg_data, delta_t, current_phase, phase)
    
    # Print queue length and arrivals for the phase
    # print("SCORES FOR PHASE: ", phase)
    # print("QUEUE: ", queue_length)
    # print("ARRIVALS: ", arrivals)
    
    
    # We are in the current green phase
    if phase == current_phase:
        departures_stay = estimate_departures1(current_phase_leg_data, delta_t, saturation_flow_rate, 'c1', yellow_time, current_phase, phase)
        departures_switch = estimate_departures1(current_phase_leg_data, delta_t, saturation_flow_rate, 'switch', yellow_time, current_phase, phase)
    
    # We are the next up phase to be green
    elif phase == (current_phase+1) % 4:
        departures_stay = estimate_departures1(current_phase_leg_data, delta_t, saturation_flow_rate, 'stay', yellow_time, current_phase, phase)
        departures_switch = estimate_departures1(current_phase_leg_data, delta_t, saturation_flow_rate, 'c2', yellow_time, current_phase, phase)
    
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

def compute_pareto_solution1(leg_data, current_phase, delta_t, saturation_flow_rate, yellow_time, min_green_time, max_green_time, green_durations, stay_will):
   
    # Check for minimum and maximum green time constraints
    if green_durations[current_phase] < min_green_time:
        return current_phase, False  # Stay in current phase
    
    if green_durations[current_phase] >= max_green_time:
        return (current_phase+1) % 4, True  # Force switch to next phase
    
    
    phases = [0, 1, 2, 3]  
    
    # Compute the payoffs for the current phase
    stay_current, switch_current = calculate_payoff1(current_phase, 
                                                    leg_data, 
                                                    delta_t, 
                                                    saturation_flow_rate, 
                                                    yellow_time, 
                                                    current_phase)
    

    # Compute it for the next phase
    next_phase = (current_phase + 1) % len(phases)
    stay_next, switch_next = calculate_payoff1(current_phase, 
                                              leg_data, 
                                              delta_t, 
                                              saturation_flow_rate, 
                                              yellow_time, 
                                              next_phase)
    
    # Compute it for the other phases
    other_1 = (current_phase + 2) % len(phases)
    other_2 = (current_phase + 3) % len(phases)
    
    stay_next_1, switch_next_1 = calculate_payoff1(current_phase, 
                                                  leg_data, 
                                                  delta_t, 
                                                  saturation_flow_rate, 
                                                  yellow_time, 
                                                  other_1)
    
    stay_next_2, switch_next_2 = calculate_payoff1(current_phase, 
                                                  leg_data, 
                                                  delta_t, 
                                                  saturation_flow_rate, 
                                                  yellow_time, 
                                                  other_2)
    
    
    # Combine next, 1 and 2 in a single coorporative player
    R_Pc_next = 0.6*stay_next + 0.3*stay_next_1 + 0.1*stay_next_2
    R_Pc_switch = 0.6*switch_next + 0.6*switch_next_1 + 0.6*switch_next_2
    
    stay_stay = 0.8*stay_current + 0.2*R_Pc_next
    switch_switch = 0.8*switch_current + 0.2*R_Pc_switch
    
    # print("FINAL PAYOFFS")
    # print("STAY_STAY: ", stay_stay)
    # print("SWITCH_SWITCH: ", switch_switch)
    
    
    if stay_will*stay_stay > (1-stay_will)*switch_switch:
        return current_phase, False
    else:
        return next_phase, True


############# FOR MAP 2 ##################


def extract_data_for_optimization2(request: TrafficSimulationPredictRequestDto):
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

def estimate_acceleration_by_distance2(speed, distance, current_phase, phase):
    
    
    if speed == 0:
        return 0
    
    elif current_phase == phase:
        return 2.6
    elif current_phase != phase:
        return -4.6
    
def estimate_queue_length2(leg_data, vth, dth):
    
    car_speeds = leg_data['speeds']
    car_distances = leg_data['distances']
    
    return sum(1 for v, d in zip(car_speeds, car_distances) if v < vth and d < dth)

def estimate_arrivals2(leg_data, delta_t, current_phase, phase):
   
    
    speeds = leg_data['speeds']  # List of vehicle speeds in leg l
    distances = leg_data['distances']  # List of vehicle distances to stop line in leg l
    #accelerations = [estimate_acceleration_by_distance(d_i, v_i) for v_i, d_i in zip(speeds, distances)]
    accelerations = [estimate_acceleration_by_distance2(speed, distance, current_phase, phase) for speed, distance in zip(speeds, distances)]
    #accelerations = [2.6] * len(speeds) if current_phase == phase else [-4.6]*len(speeds)
    
    
    arrivals = []
    for v_i, a_i in zip(speeds, accelerations):
        if a_i == 0:
            a_i = 0.01  # Avoid division by zero
        arrivals.append(np.sign(max(0, delta_t - v_i / a_i)))  # Sign function based on time to stop

    A_l = sum(arrivals)
    
    return A_l

def estimate_departures2(leg_data, delta_t, saturation_flow_rate, current_phase_condition, yellow_time, current_phase, phase):
   
   
    # Condition c1: Use saturation flow rate
    if current_phase_condition == 'c1':
        return saturation_flow_rate * delta_t
    
    # Condition c2: The next phase to be green
    elif current_phase_condition == 'c2':
        speeds = leg_data['speeds']
        distances = leg_data['distances']
        #accelerations = [estimate_acceleration_by_distance(d_i, v_i) for v_i, d_i in zip(speeds, distances)]
        #accelerations = [2.6] * len(speeds) if current_phase == phase else [-4.6]*len(speeds)
        accelerations = [estimate_acceleration_by_distance2(speed, distance, current_phase, phase) for speed, distance in zip(speeds, distances)]

        # Compute ΔL for each vehicle in the leg
        departures = []
        for v_i, d_i, a_i in zip(speeds, distances, accelerations):
            delta_L = v_i * (delta_t - yellow_time) + 0.5 * a_i * (delta_t - yellow_time)**2 - d_i
            #print("DELTA L: ", delta_L)
            departure = np.sign(max(0, -delta_L))  # Sign function based on ΔL
            departures.append(departure)
        
        # Sum the number of departures for all vehicles
        D_l = sum(departures)
        #print("------- GOT IN HEEEERE----- AND GOT: ", D_l)
        return D_l
    
    # No departing vehicles no matter what
    else: 
        return 0

def calculate_payoff2(current_phase, leg_data, delta_t, saturation_flow_rate, yellow_time, phase):
    
    # Current phase is the phase which is currently green
    # Phase is the phase which we want to calculate the payoff for
   
    # Extract the speed and distances for the current phase
    current_phase_leg_data = list(leg_data.values())[phase]
    queue_length = estimate_queue_length2(current_phase_leg_data, 1, 100)
    arrivals = estimate_arrivals2(current_phase_leg_data, delta_t, current_phase, phase)
    
    # Print queue length and arrivals for the phase
    #print("SCORES FOR PHASE: ", phase)
    #print("QUEUE: ", queue_length)
    #print("ARRIVALS: ", arrivals)
    
    
    # We are in the current green phase
    if phase == current_phase:
        departures_stay = estimate_departures2(current_phase_leg_data, delta_t, saturation_flow_rate, 'c1', yellow_time, current_phase, phase)
        departures_switch = estimate_departures2(current_phase_leg_data, delta_t, saturation_flow_rate, 'switch', yellow_time, current_phase, phase)
    
    # We are the next up phase to be green
    elif phase == (current_phase+1) % 3:
        departures_stay = estimate_departures2(current_phase_leg_data, delta_t, saturation_flow_rate, 'stay', yellow_time, current_phase, phase)
        departures_switch = estimate_departures2(current_phase_leg_data, delta_t, saturation_flow_rate, 'c2', yellow_time, current_phase, phase)
    
    # We are in the other phases - Will get no departures no matter what
    else:
        departures_stay = 0
        departures_switch = 0
    
    
    #print("DEPARTURES (STAY): ", max(departures_stay, departures_switch))
    #print("DEPARTURES (SWITCH): ", min(departures_stay, departures_switch))
    
    #print("PAYOFF (STAY): ", queue_length + arrivals - departures_stay)
    #print("PAYOFF (SWITCH): ", queue_length + arrivals - departures_switch)
    #print()
    reward_stay = queue_length + arrivals - departures_stay
    reward_switch = queue_length + arrivals - departures_switch

    return reward_stay, reward_switch

def compute_pareto_solution2(leg_data, current_phase, delta_t, saturation_flow_rate, yellow_time, min_green_time, max_green_time, green_durations, stay_will=0.6):
   
    # Check for minimum and maximum green time constraints
    if green_durations[current_phase] < min_green_time:
        return current_phase, False  # Stay in current phase
    
    if green_durations[current_phase] >= max_green_time:
        return (current_phase+1) % 3, True  # Force switch to next phase
    
    
    phases = [0, 1, 2]  
    
    # Compute the payoffs for the current phase
    stay_current, switch_current = calculate_payoff2(current_phase, 
                                                    leg_data, 
                                                    delta_t, 
                                                    saturation_flow_rate, 
                                                    yellow_time, 
                                                    current_phase)
    

    # Compute it for the next phase
    next_phase = (current_phase + 1) % len(phases)
    stay_next, switch_next = calculate_payoff2(current_phase, 
                                              leg_data, 
                                              delta_t, 
                                              saturation_flow_rate, 
                                              yellow_time, 
                                              next_phase)
    
    # Compute it for the other phases
    other_1 = (current_phase + 2) % len(phases)

    
    stay_next_1, switch_next_1 = calculate_payoff2(current_phase, 
                                                  leg_data, 
                                                  delta_t, 
                                                  saturation_flow_rate, 
                                                  yellow_time, 
                                                  other_1)
    
    
    # Combine next, 1 and 2 in a single coorporative player
    R_Pc_next = 0.6*stay_next + 0.4*stay_next_1 
    R_Pc_switch = 0.6*switch_next + 0.6*switch_next_1 
    
    stay_stay = 0.8*stay_current + 0.2*R_Pc_next
    switch_switch = 0.8*switch_current + 0.2*R_Pc_switch
    
    #print("FINAL PAYOFFS")
    #print("STAY_STAY: ", stay_stay)
    #print("SWITCH_SWITCH: ", switch_switch)
    
    
    if stay_will*stay_stay > (1-stay_will)*switch_switch:
        return current_phase, False
    else:
        return next_phase, True

#######################################











################################################


global signal_state_durations, active_group_durations, changed_map
# changed_map = True
changed_map = False

signal_state_durations = defaultdict(lambda: defaultdict(int))
active_group_durations = defaultdict(int)
# Parameters for both maps
current_map = 1  # Start with map 1
tick_count = 0

# Parameters for Map 1
current_phase = 0  # Start with phase P1
num_phases = 4  # Number of phases in the signal cycle
delta_t = 5.1
saturation_flow_rate = 1.6  # Vehicles per second per leg
yellow_time = 4
min_green_time = 6
max_green_time = 30
stay_will = 0.6


# Overlap duration in ticks (assuming 1 tick corresponds to 1 second)
OVERLAP_DURATION_TICKS = 1
overlap_ticks_remaining = 0  # Initialize overlap ticks remaining
next_phase = current_phase  # Initialize next_phase

green_durations = [0] * num_phases  # Track how long each phase has been green
phase_signals = {
    0: ['A1', 'A1LeftTurn'],  # Phase 1
    1: ['A2', 'A2LeftTurn'],  # Phase 2
    2: ['B1', 'B1LeftTurn'],  # Phase 3
    3: ['B2', 'B2LeftTurn']   # Phase 4
}


app = FastAPI()

@app.get('/api')
def hello():
    return {
        "service": "traffic-simulation-usecase",
        "uptime": '{}'.format(datetime.timedelta(seconds=tick_count))
    }

@app.get('/')
def index():
    return "Your endpoint is running!"




@app.post('/predict', response_model=TrafficSimulationPredictResponseDto)
def predict_endpoint(request: TrafficSimulationPredictRequestDto):
    global current_map, tick_count
    global current_phase, next_phase, green_durations, delta_t, saturation_flow_rate, yellow_time, min_green_time, max_green_time, phase_signals, num_phases
    global signal_state_durations, active_group_durations, changed_map
    global overlap_ticks_remaining  # Include overlap_ticks_remaining in globals

    # Decode request data
    vehicles = request.vehicles
    signals = request.signals
    legs = request.legs
    current_time = request.simulation_ticks
    tick_count = current_time

    logger.info(f"\033[94mScore at tick: {request.total_score}\033[0m")
    logger.info(f'\033[96mNumber of vehicles at tick {current_time}: {len(vehicles)}\033[0m')
    logger.info(f"Current map: {current_map}")

    # Check if it's time to switch to the second map
    if legs[2].lanes == ['Main', 'Right'] and not changed_map:
        logger.warning("\033[93mReset detected! Switching to the second signal list.\033[0m")
        current_map = 2
        # Reset variables for map 2
        num_phases = 3
        phase_signals = {
            0: ['A1', 'A1LeftTurn', 'A1RightTurn'],  # Phase 1
            1: ['A2', 'A2LeftTurn', 'A2RightTurn'],  # Phase 2
            2: ['B1', 'B1RightTurn', 'B2']           # Phase 3
        }
        green_durations = [0] * num_phases
        current_phase = 0  # Reset current_phase for Map 2
        next_phase = current_phase
        overlap_ticks_remaining = 0  # Reset overlap
        changed_map = True

    if current_map == 1:
        # Map 1 logic with overlap implemented
        leg_data = extract_data_for_optimization1(request)

        # Call compute_pareto_solution1
        next_phase_decision, should_switch = compute_pareto_solution1(
            leg_data,
            current_phase,
            delta_t,
            saturation_flow_rate,
            yellow_time,
            min_green_time,
            max_green_time,
            green_durations,
            stay_will
        )

        # Overlap logic
        if overlap_ticks_remaining > 0:
            # During overlap, keep both current and next phase signals green
            overlap_ticks_remaining -= 1
            logger.info(f"Overlap in progress. Ticks remaining: {overlap_ticks_remaining}")
            green_signal_group = phase_signals[current_phase] + phase_signals[next_phase]
            if overlap_ticks_remaining == 0:
                # Overlap finished, switch to next phase
                current_phase = next_phase
                green_durations = [0] * num_phases  # Reset green durations
                logger.info(f"Transition to phase {current_phase} complete.")
        else:
            if should_switch:
                # Start overlap
                next_phase = next_phase_decision
                overlap_ticks_remaining = OVERLAP_DURATION_TICKS
                logger.info(f"Starting overlap between phase {current_phase} and phase {next_phase}.")
                green_signal_group = phase_signals[current_phase] + phase_signals[next_phase]
            else:
                # Continue with current phase
                green_durations[current_phase] += 1
                green_signal_group = phase_signals[current_phase]

        # Prepare the response with the correct signals
        all_signals = [signal.name for signal in signals]

        # Count vehicles based on the leg's signal groups
        signal_group_vehicle_counts = defaultdict(int)
        for vehicle in vehicles:
            for leg in legs:
                if vehicle.leg == leg.name:
                    for signal_group in leg.signal_groups:
                        signal_group_vehicle_counts[signal_group] += 1

        # Set the signals to green if they are in the green_signal_group, otherwise red
        next_signals = []
        for signal in signals:
            # Log signal state durations
            if signal.state == "green":
                signal_state_durations[signal.name]["green"] += 1
            elif signal.state == "red":
                signal_state_durations[signal.name]["red"] += 1
            elif signal.state == "amber":
                signal_state_durations[signal.name]["amber"] += 1
            elif signal.state == "redamber":
                signal_state_durations[signal.name]["redamber"] += 1

            # Track how long the signal has been part of the active group
            if signal.name in green_signal_group:
                active_group_durations[signal.name] += 1

            # Update signal state for the next tick
            if signal.name in green_signal_group:
                next_signals.append(SignalDto(name=signal.name, state="green"))
                logger.info(f"\033[92mSignal {signal.name} is {signal.state} (in active group) with {signal_group_vehicle_counts.get(signal.name, 0)} vehicles.\033[0m")
            else:
                next_signals.append(SignalDto(name=signal.name, state="red"))
                logger.info(f"\033[91mSignal {signal.name} is {signal.state} (not in active group) with {signal_group_vehicle_counts.get(signal.name, 0)} vehicles.\033[0m")

        # Log the current phase and active signal group
        logger.info(f"\033[94mCurrent phase: {current_phase}\033[0m")
        logger.info(f"\033[93mActive signal group: {green_signal_group}\033[0m")

        # Update the text file silently after each tick
        log_results_to_file(current_map)

        # Return the updated signals to the simulation
        response = TrafficSimulationPredictResponseDto(signals=next_signals)
        return response
    else:
        # Map 2 logic with overlap implemented
        leg_data = extract_data_for_optimization2(request)

        # Call compute_pareto_solution2
        next_phase_decision, should_switch = compute_pareto_solution2(
            leg_data,
            current_phase,
            delta_t,
            saturation_flow_rate,
            yellow_time,
            min_green_time,
            max_green_time,
            green_durations,
            stay_will
        )

        # Overlap logic for Map 2
        if overlap_ticks_remaining > 0:
            # During overlap, keep both current and next phase signals green
            overlap_ticks_remaining -= 1
            logger.info(f"Overlap in progress. Ticks remaining: {overlap_ticks_remaining}")
            green_signal_group = phase_signals[current_phase] + phase_signals[next_phase]
            if overlap_ticks_remaining == 0:
                # Overlap finished, switch to next phase
                current_phase = next_phase
                green_durations = [0] * num_phases  # Reset green durations
                logger.info(f"Transition to phase {current_phase} complete.")
        else:
            if should_switch:
                # Start overlap
                next_phase = next_phase_decision
                overlap_ticks_remaining = OVERLAP_DURATION_TICKS
                logger.info(f"Starting overlap between phase {current_phase} and phase {next_phase}.")
                green_signal_group = phase_signals[current_phase] + phase_signals[next_phase]
            else:
                # Continue with current phase
                green_durations[current_phase] += 1
                green_signal_group = phase_signals[current_phase]

        # Prepare the response with the correct signals
        all_signals = [signal.name for signal in signals]

        # Count vehicles based on the leg's signal groups
        signal_group_vehicle_counts = defaultdict(int)
        for vehicle in vehicles:
            for leg in legs:
                if vehicle.leg == leg.name:
                    for signal_group in leg.signal_groups:
                        signal_group_vehicle_counts[signal_group] += 1

        # Set the signals to green if they are in the green_signal_group, otherwise red
        next_signals = []
        for signal in signals:
            # Log signal state durations
            if signal.state == "green":
                signal_state_durations[signal.name]["green"] += 1
            elif signal.state == "red":
                signal_state_durations[signal.name]["red"] += 1
            elif signal.state == "amber":
                signal_state_durations[signal.name]["amber"] += 1
            elif signal.state == "redamber":
                signal_state_durations[signal.name]["redamber"] += 1

            # Track how long the signal has been part of the active group
            if signal.name in green_signal_group:
                active_group_durations[signal.name] += 1

            # Update signal state for the next tick
            if signal.name in green_signal_group:
                next_signals.append(SignalDto(name=signal.name, state="green"))
                logger.info(f"\033[92mSignal {signal.name} is {signal.state} (in active group) with {signal_group_vehicle_counts.get(signal.name, 0)} vehicles.\033[0m")
            else:
                next_signals.append(SignalDto(name=signal.name, state="red"))
                logger.info(f"\033[91mSignal {signal.name} is {signal.state} (not in active group) with {signal_group_vehicle_counts.get(signal.name, 0)} vehicles.\033[0m")

        # Log the current phase and active signal group
        logger.info(f"\033[94mCurrent phase: {current_phase}\033[0m")
        logger.info(f"\033[93mActive signal group: {green_signal_group}\033[0m")

        # Update the text file silently after each tick
        log_results_to_file(current_map)

        # Return the updated signals to the simulation
        response = TrafficSimulationPredictResponseDto(signals=next_signals)
        return response

def log_results_to_file(map_number):
    filename = f"traffic_log_map_{map_number}.txt"
    with open(filename, "w") as f:
        f.write(f"Results for map {map_number} at tick {tick_count}:\n")
        f.write("Signal state durations (in ticks):\n")
        for signal, states in signal_state_durations.items():
            f.write(f"Signal {signal}: {states}\n")
        f.write("\nActive group durations (in ticks):\n")
        for signal, ticks in active_group_durations.items():
            f.write(f"Signal {signal}: {ticks} ticks in active group\n")

if __name__ == '__main__':
    uvicorn.run(
        'api:app',
        host="0.0.0.0",
        port=8080
    )