from multiprocessing import Process, Queue
from time import sleep, time
from environment import load_and_run_simulation
import numpy as np
from typing import List, Dict

# Define the overlap duration in ticks (assuming 1 tick corresponds to 1 second)
OVERLAP_DURATION_TICKS = 1

##########################################
# Helper functions for traffic signal optimization
##########################################

def extract_data_for_optimization(request):
    """
    Extract vehicle speeds and distances for each leg in the simulation.
    
    Args:
        request (TrafficSimulationPredictRequestDto): The current state of the simulation.
    
    Returns:
        Dict: A dictionary with leg names as keys, containing lists of vehicle speeds and distances.
    """
    # Initialize dictionary to hold vehicle data (speeds and distances) for each leg
    leg_data = {leg.name: {'speeds': [], 'distances': []} for leg in request.legs}

    # Accumulate data for each vehicle in its corresponding leg
    for vehicle in request.vehicles:
        leg_name = vehicle.leg  # The leg the vehicle belongs to
        leg_data[leg_name]['speeds'].append(vehicle.speed)
        leg_data[leg_name]['distances'].append(vehicle.distance_to_stop)

    return leg_data

def estimate_acceleration_by_distance(speed, distance, current_phase, phase):
    """
    Estimate vehicle acceleration based on its speed and distance to stop.
    
    Args:
        speed (float): The speed of the vehicle.
        distance (float): The distance of the vehicle to the stop line.
        current_phase (int): The current phase of the traffic signal.
        phase (int): The phase under evaluation.
    
    Returns:
        float: Estimated acceleration based on the current phase.
    """
    if speed == 0:
        return 0
    elif current_phase == phase:
        return 2.6  # Positive acceleration if in the same phase
    else:
        return -4.6  # Deceleration if changing phases

def estimate_queue_length(leg_data, vth, dth):
    """
    Estimate the number of queued vehicles based on speed and distance thresholds.
    
    Args:
        leg_data (Dict): Vehicle data containing speeds and distances for a leg.
        vth (float): Speed threshold for considering a vehicle as queued.
        dth (float): Distance threshold for considering a vehicle as queued.
    
    Returns:
        int: Estimated number of queued vehicles.
    """
    car_speeds = leg_data['speeds']
    car_distances = leg_data['distances']
    return sum(1 for v, d in zip(car_speeds, car_distances) if v < vth and d < dth)

def estimate_arrivals(leg_data, delta_t, current_phase, phase):
    """
    Estimate the number of vehicle arrivals at the intersection.
    
    Args:
        leg_data (Dict): Vehicle data for a leg.
        delta_t (float): Time interval for estimation.
        current_phase (int): The current signal phase.
        phase (int): The phase under evaluation.
    
    Returns:
        int: Estimated number of vehicle arrivals.
    """
    speeds = leg_data['speeds']
    distances = leg_data['distances']
    accelerations = [estimate_acceleration_by_distance(speed, distance, current_phase, phase) for speed, distance in zip(speeds, distances)]
    
    arrivals = []
    for v_i, a_i in zip(speeds, accelerations):
        if a_i == 0:
            a_i = 0.01  # Avoid division by zero
        arrivals.append(np.sign(max(0, delta_t - v_i / a_i)))
    
    return sum(arrivals)

def estimate_departures(leg_data, delta_t, saturation_flow_rate, current_phase_condition, yellow_time, current_phase, phase):
    """
    Estimate the number of vehicle departures during the signal phase.
    
    Args:
        leg_data (Dict): Vehicle data for a leg.
        delta_t (float): Time interval for estimation.
        saturation_flow_rate (float): Maximum flow rate.
        current_phase_condition (str): Condition of the current signal phase (e.g., 'c1', 'c2', etc.).
        yellow_time (float): Duration of the yellow signal.
        current_phase (int): The current signal phase.
        phase (int): The phase under evaluation.
    
    Returns:
        int: Estimated number of vehicle departures.
    """
    if current_phase_condition == 'c1':
        return saturation_flow_rate * delta_t
    elif current_phase_condition == 'c2':
        speeds = leg_data['speeds']
        distances = leg_data['distances']
        accelerations = [estimate_acceleration_by_distance(speed, distance, current_phase, phase) for speed, distance in zip(speeds, distances)]
        
        departures = []
        for v_i, d_i, a_i in zip(speeds, distances, accelerations):
            delta_L = v_i * (delta_t - yellow_time) + 0.5 * a_i * (delta_t - yellow_time)**2 - d_i
            departure = np.sign(max(0, -delta_L))
            departures.append(departure)
        
        return sum(departures)
    else:
        return 0

def calculate_payoff(current_phase, leg_data, delta_t, saturation_flow_rate, yellow_time, phase):
    """
    Calculate the payoff (reward) for staying in or switching from the current signal phase.
    
    Args:
        current_phase (int): The current signal phase.
        leg_data (Dict): Vehicle data for all legs.
        delta_t (float): Time interval for estimation.
        saturation_flow_rate (float): Maximum flow rate.
        yellow_time (float): Duration of the yellow signal.
        phase (int): The phase under evaluation.
    
    Returns:
        Tuple: The rewards for staying in the current phase and switching to the next phase.
    """
    current_phase_leg_data = list(leg_data.values())[phase]
    queue_length = estimate_queue_length(current_phase_leg_data, 1, 100)
    arrivals = estimate_arrivals(current_phase_leg_data, delta_t, current_phase, phase)
    
    if phase == current_phase:
        departures_stay = estimate_departures(current_phase_leg_data, delta_t, saturation_flow_rate, 'c1', yellow_time, current_phase, phase)
        departures_switch = estimate_departures(current_phase_leg_data, delta_t, saturation_flow_rate, 'switch', yellow_time, current_phase, phase)
    elif phase == (current_phase + 1) % 4:
        departures_stay = estimate_departures(current_phase_leg_data, delta_t, saturation_flow_rate, 'stay', yellow_time, current_phase, phase)
        departures_switch = estimate_departures(current_phase_leg_data, delta_t, saturation_flow_rate, 'c2', yellow_time, current_phase, phase)
    else:
        departures_stay = 0
        departures_switch = 0
    
    reward_stay = queue_length + arrivals - departures_stay
    reward_switch = queue_length + arrivals - departures_switch
    
    return reward_stay, reward_switch

def compute_pareto_solution(leg_data, current_phase, delta_t, saturation_flow_rate, yellow_time, min_green_time, max_green_time, green_durations, stay_will=0.6):
    """
    Compute the Pareto-optimal solution for whether to stay in the current phase or switch to the next.
    
    Args:
        leg_data (Dict): Vehicle data for all legs.
        current_phase (int): The current signal phase.
        delta_t (float): Time interval for estimation.
        saturation_flow_rate (float): Maximum flow rate.
        yellow_time (float): Duration of the yellow signal.
        min_green_time (int): Minimum green time for a phase.
        max_green_time (int): Maximum green time for a phase.
        green_durations (List[int]): List of green durations for each phase.
        stay_will (float): Weight factor for staying in the current phase.
    
    Returns:
        Tuple: The next phase and a boolean indicating if the switch should happen.
    """
    if green_durations[current_phase] < min_green_time:
        return current_phase, False  # Stay in current phase
    if green_durations[current_phase] >= max_green_time:
        return (current_phase + 1) % 4, True  # Force switch to next phase
    
    phases = [0, 1, 2, 3]
    stay_current, switch_current = calculate_payoff(current_phase, leg_data, delta_t, saturation_flow_rate, yellow_time, current_phase)
    next_phase = (current_phase + 1) % len(phases)
    
    stay_next, switch_next = calculate_payoff(current_phase, leg_data, delta_t, saturation_flow_rate, yellow_time, next_phase)
    
    other_1 = (current_phase + 2) % len(phases)
    other_2 = (current_phase + 3) % len(phases)
    
    stay_next_1, switch_next_1 = calculate_payoff(current_phase, leg_data, delta_t, saturation_flow_rate, yellow_time, other_1)
    stay_next_2, switch_next_2 = calculate_payoff(current_phase, leg_data, delta_t, saturation_flow_rate, yellow_time, other_2)
    
    R_Pc_next = 0.6 * stay_next + 0.3 * stay_next_1 + 0.1 * stay_next_2
    R_Pc_switch = 0.6 * switch_next + 0.6 * switch_next_1 + 0.6 * switch_next_2
    
    stay_stay = 0.8 * stay_current + 0.2 * R_Pc_next
    switch_switch = 0.8 * switch_current + 0.2 * R_Pc_switch
    
    if stay_will * stay_stay > (1 - stay_will) * switch_switch:
        return current_phase, False
    else:
        return next_phase, True

##########################################
# Main simulation function
##########################################

def run_game():
    """
    Main function to run the traffic simulation and control signal phases.
    """
    test_duration_seconds = 300  # Total simulation time in seconds
    random = True
    configuration_file = "models/1/glue_configuration.yaml"  # Configuration file for simulation
    start_time = time()

    input_queue = Queue()  # Queue for sending signal states to the simulation
    output_queue = Queue()  # Queue for receiving state updates from the simulation
    error_queue = Queue()  # Queue for handling errors

    # Start the simulation in a separate process
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
    current_phase = 0  # Start with phase 1
    delta_t = 6  # Time step for phase changes
    saturation_flow_rate = 1.1  # Maximum flow rate
    yellow_time = 3  # Duration of yellow signal
    min_green_time = 6  # Minimum green time for a phase
    max_green_time = 30  # Maximum green time for a phase
    stay_will = 0.6  # Weight factor for staying in the current phase

    # Define the signal groups for each phase
    phase_signals = {
        0: ['A1', 'A1LeftTurn'],  # Phase 1
        1: ['A2', 'A2LeftTurn'],  # Phase 2
        2: ['B1', 'B1LeftTurn'],  # Phase 3
        3: ['B2', 'B2LeftTurn']   # Phase 4
    }

    green_durations = [0] * num_phases  # Track how long each phase has been green
    overlap_ticks_remaining = 0  # Track overlap duration between phase switches
    next_phase = current_phase  # Initialize next phase

    while True:
        # Get the current simulation state from the output queue
        state = output_queue.get()

        if state.is_terminated:
            break  # Exit the loop if the simulation is finished

        # Extract vehicle data from the simulation state
        leg_data = extract_data_for_optimization(state)

        # Compute whether to stay in the current phase or switch to the next
        next_phase_decision, should_switch = compute_pareto_solution(leg_data,
                                                                     current_phase,
                                                                     delta_t,
                                                                     saturation_flow_rate,
                                                                     yellow_time,
                                                                     min_green_time,
                                                                     max_green_time,
                                                                     green_durations,
                                                                     stay_will)

        if overlap_ticks_remaining > 0:
            # During overlap, both current and next phase signals stay green
            overlap_ticks_remaining -= 1
            green_signal_group = phase_signals[current_phase] + phase_signals[next_phase]
            if overlap_ticks_remaining == 0:
                # Once overlap is done, switch to the next phase
                current_phase = next_phase
                green_durations = [0] * num_phases  # Reset green durations
        else:
            if should_switch:
                # Start the overlap between current and next phase
                next_phase = next_phase_decision
                overlap_ticks_remaining = OVERLAP_DURATION_TICKS
                green_signal_group = phase_signals[current_phase] + phase_signals[next_phase]
            else:
                # Continue with the current phase
                green_durations[current_phase] += 1
                green_signal_group = phase_signals[current_phase]

        # Update signal states for the simulation
        all_signals = state.signal_groups
        response_signals = {signal_group: "green" if signal_group in green_signal_group else "red" for signal_group in all_signals}

        # Send the updated signals to the simulation
        input_queue.put(response_signals)

    print(f"Simulation complete. Total score: {state.total_score}")

if __name__ == '__main__':
    run_game()
