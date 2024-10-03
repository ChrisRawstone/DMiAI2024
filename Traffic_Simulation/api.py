import uvicorn
from fastapi import FastAPI
import datetime
from loguru import logger
from pydantic import BaseModel
from sim.dtos import TrafficSimulationPredictResponseDto, TrafficSimulationPredictRequestDto, SignalDto
from collections import defaultdict, deque
from multiprocessing import Queue

# Import functions from your provided module
from sim.pareto import (
    extract_data_for_optimization,
    estimate_acceleration_by_distance,
    estimate_queue_length,
    estimate_arrivals,
    estimate_departures,
    calculate_payoff,
    compute_pareto_solution
)

# Parameters for both maps
current_map = 1  # Start with map 1
tick_count = 0

# Parameters for Map 1
current_phase = 0  # Start with phase P1
num_phases = 4  # Number of phases in the signal cycle
delta_t = 6
saturation_flow_rate = 1.1  # Vehicles per second per leg
yellow_time = 3
min_green_time = 6
max_green_time = 30
green_durations = [0] * num_phases  # Track how long each phase has been green
phase_signals = {
    0: ['A1', 'A1LeftTurn'],  # Phase 1
    1: ['A2', 'A2LeftTurn'],  # Phase 2
    2: ['B1', 'B1LeftTurn'],  # Phase 3
    3: ['B2', 'B2LeftTurn']   # Phase 4
}

# Parameters for Map 2
WINDOW_SIZE = 5  # Number of ticks over which we calculate the change
PROLONG_TICKS = 5  # Prolong the green period by 5 ticks if waiting cars decrease significantly
REDUCE_TICKS = 15  # Reduce the green period by 15 ticks if waiting cars increase
speed_threshold = 0.05  # Speed threshold to consider a vehicle as waiting

# Logging variables for signal state durations and active group durations
signal_state_durations = defaultdict(lambda: defaultdict(int))
active_group_durations = defaultdict(int)

# Traffic light parameters
VEHICLE_THRESHOLD = 5  # Vehicle count to trigger reset

# List for signal combinations for Map 2
second_list = [
    ['A1', 'A1RightTurn', 'A1LeftTurn'],
    ['A2', 'A2RightTurn', 'A2LeftTurn'],
    ['B1', 'B1RightTurn', 'B2']
]

second_original_timer = [25, 25, 25]  # Original green light durations for the second map
second_timer = second_original_timer.copy()  # Temporary green light durations (adjusted per cycle)

# Variables to track previous waiting vehicle counts for adaptive adjustments
waiting_counts_history = deque(maxlen=WINDOW_SIZE)
adjustment_made = False

# Initial setup for Map 2
current_list = second_list
current_timer = second_original_timer.copy()  # Start with second timer
current_index = 0
last_switch_tick = 0  # Use ticks to track when the last switch happened

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
    global current_phase, green_durations, delta_t, saturation_flow_rate, yellow_time, min_green_time, max_green_time, phase_signals, num_phases
    global current_list, current_timer, current_index, last_switch_tick, adjustment_made, waiting_counts_history
    global signal_state_durations, active_group_durations

    # Decode request data
    vehicles = request.vehicles
    signals = request.signals
    legs = request.legs
    current_time = request.simulation_ticks
    tick_count = current_time

    logger.info(f"\033[94mScore at tick: {request.total_score}\033[0m")
    logger.info(f'\033[96mNumber of vehicles at tick {current_time}: {len(vehicles)}\033[0m')

    # Check if it's time to switch to the second map based on tick count and vehicle count
    if tick_count > 200 and len(vehicles) < VEHICLE_THRESHOLD and current_map == 1:
        logger.warning("\033[93mReset detected! Switching to the second signal list.\033[0m")
        current_map = 2
        # Reset variables for map 2
        current_list = second_list
        current_timer = second_original_timer.copy()
        current_index = 0
        last_switch_tick = current_time
        adjustment_made = False
        waiting_counts_history.clear()
        signal_state_durations = defaultdict(lambda: defaultdict(int))
        active_group_durations = defaultdict(int)

    if current_map == 1:
        # Map 1 logic using the imported functions
        leg_data = extract_data_for_optimization(request)

        # Call compute_pareto_solution
        next_phase, should_switch = compute_pareto_solution(
            leg_data,
            current_phase,
            delta_t,
            saturation_flow_rate,
            yellow_time,
            min_green_time,
            max_green_time,
            green_durations
        )

        if should_switch:
            current_phase = next_phase
            green_durations = [0] * num_phases  # Reset all green durations when switching
        else:
            green_durations[current_phase] += 1  # Increment time for the current green phase

        # Prepare the response with the correct signals for the new phase
        green_signal_group = phase_signals[current_phase]  # Get the signal group for the current phase
        all_signals = [signal.name for signal in signals]  # List of all signal group names

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
        # Map 2 logic (existing logic)
        leg_vehicle_counts = defaultdict(int)
        waiting_vehicle_counts = defaultdict(int)

        # Count vehicles and waiting vehicles (waiting if speed is 0 or near 0)
        for vehicle in vehicles:
            leg_vehicle_counts[vehicle.leg] += 1
            if vehicle.speed <= speed_threshold:
                waiting_vehicle_counts[vehicle.leg] += 1

        # Store the current tick's waiting vehicle counts in history
        waiting_counts_history.append(waiting_vehicle_counts.copy())

        # Log the waiting vehicle counts and their changes
        if len(waiting_counts_history) == WINDOW_SIZE:
            previous_waiting_counts = waiting_counts_history[0]
            change_in_waiting_counts = {leg: waiting_vehicle_counts[leg] - previous_waiting_counts.get(leg, 0) for leg in waiting_vehicle_counts}
            logger.info(f"\033[94mChange in waiting vehicles for the last {WINDOW_SIZE} ticks: {change_in_waiting_counts}\033[0m")

            # Apply adaptive logic continuously
            active_group_legs = current_list[current_index]
            active_changes = [change_in_waiting_counts.get(leg, 0) for leg in active_group_legs]

            if all(change < -4 for change in active_changes) and not adjustment_made:
                logger.info(f"\033[92mProlonging green period for {active_group_legs} by {PROLONG_TICKS} ticks.\033[0m")
                current_timer[current_index] += PROLONG_TICKS
                adjustment_made = True
            elif all(change > -1 for change in active_changes) and not adjustment_made:
                logger.info(f"\033[91mReducing green period for {active_group_legs} by {REDUCE_TICKS} ticks.\033[0m")
                current_timer[current_index] = max(1, current_timer[current_index] - REDUCE_TICKS)
                adjustment_made = True

        # Get the green light duration for the current signal group based on ticks
        green_light_duration = current_timer[current_index]
        elapsed_ticks = current_time - last_switch_tick
        remaining_ticks = green_light_duration - elapsed_ticks

        logger.info(f"\033[95mRemaining ticks for current signal group: {remaining_ticks} ticks\033[0m")

        if elapsed_ticks >= green_light_duration:
            current_index = (current_index + 1) % len(current_list)
            logger.info(f"\033[93mSwitching to signal group: {current_list[current_index]}\033[0m")
            last_switch_tick = current_time
            current_timer = second_original_timer.copy()
            adjustment_made = False

        logger.info(f"\033[94mSecond list is used\033[0m")
        active_group = current_list[current_index]
        logger.info(f"\033[93mActive signal group: {active_group}\033[0m")

        next_signals = []
        signal_group_vehicle_counts = defaultdict(int)

        for vehicle in vehicles:
            for leg in legs:
                if vehicle.leg == leg.name:
                    for signal_group in leg.signal_groups:
                        signal_group_vehicle_counts[signal_group] += 1

        for signal in signals:
            if signal.state == "green":
                signal_state_durations[signal.name]["green"] += 1
            elif signal.state == "red":
                signal_state_durations[signal.name]["red"] += 1
            elif signal.state == "amber":
                signal_state_durations[signal.name]["amber"] += 1
            elif signal.state == "redamber":
                signal_state_durations[signal.name]["redamber"] += 1

            if signal.name in active_group:
                active_group_durations[signal.name] += 1

            if signal.name in active_group:
                next_signals.append(SignalDto(name=signal.name, state="green"))
                logger.info(f"\033[92mSignal {signal.name} is {signal.state} (in active group) with {signal_group_vehicle_counts.get(signal.name, 0)} vehicles.\033[0m")
            else:
                next_signals.append(SignalDto(name=signal.name, state="red"))
                logger.info(f"\033[91mSignal {signal.name} is {signal.state} (not in active group) with {signal_group_vehicle_counts.get(signal.name, 0)} vehicles.\033[0m")

        # Update the text file silently after each tick
        log_results_to_file(current_map)

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
