import uvicorn
from fastapi import FastAPI
import datetime
from loguru import logger
from pydantic import BaseModel
from sim.dtos import TrafficSimulationPredictResponseDto, TrafficSimulationPredictRequestDto, SignalDto
from collections import defaultdict, deque

# Adaptive timing constants
WINDOW_SIZE = 5  # Number of ticks over which we calculate the change
PROLONG_TICKS = 5  # Prolong the green period by 5 ticks if waiting cars decrease significantly
REDUCE_TICKS = 15  # Reduce the green period by 15 ticks if waiting cars increase
speed_threshold = 0.05  # Speed threshold to consider a vehicle as waiting

# Logging variables for signal state durations and active group durations
signal_state_durations = defaultdict(lambda: defaultdict(int))
active_group_durations = defaultdict(int)

# Traffic light parameters
VEHICLE_THRESHOLD = 5  # Vehicle count to trigger reset
RESET_TICK_THRESHOLD = 50  # Check reset only after 50 ticks

# List for signal combinations
first_list = [
    ['A1', 'A1LeftTurn'],
    ['A2', 'A2LeftTurn'],
    ['B1', 'B1LeftTurn'],
    ['B2', 'B2LeftTurn']
]

# Define original timers
original_timer = [30, 20, 20, 25]  # Original green light durations for first map
current_timer = original_timer.copy()  # Temporary green light durations (adjusted per cycle)

second_list = [
    ['A1', 'A1RightTurn', 'A1LeftTurn'],
    ['A2', 'A2RightTurn', 'A2LeftTurn'],
    ['B1', 'B1RightTurn', 'A1RightTurn'],
    ['B2', 'A2RightTurn', 'B1RightTurn']
]

second_original_timer = [25, 25, 25, 25]  # Original green light durations for second map
second_timer = second_original_timer.copy()  # Temporary green light durations (adjusted per cycle)

# Variables to track previous waiting vehicle counts for adaptive adjustments
waiting_counts_history = deque(maxlen=WINDOW_SIZE)
adjustment_made = False

# Initial setup
app = FastAPI()
current_list = first_list
current_timer = original_timer.copy()  # Start with first timer
current_index = 0
last_switch_tick = 0  # Use ticks to track when the last switch happened
tick_count = 0
current_map = 1  # Track the current map (1 or 2)

@app.get('/api')
def hello():
    return {
        "service": "traffic-simulation-usecase",
        "uptime": '{}'.format(datetime.timedelta(seconds=tick_count))
    }

@app.get('/')
def index():
    return "Your endpoint is running!"

timetochange = False

@app.post('/predict', response_model=TrafficSimulationPredictResponseDto)
def predict_endpoint(request: TrafficSimulationPredictRequestDto):
    global current_list, current_timer, current_index, tick_count, last_switch_tick, timetochange, current_map
    global signal_state_durations, active_group_durations, waiting_counts_history, adjustment_made

    # Decode request data
    vehicles = request.vehicles
    signals = request.signals
    legs = request.legs  # Assuming legs contain the signal groups
    current_time = request.simulation_ticks  # Get current tick count from request
    tick_count = current_time  # Track the current tick count

    logger.info(f'Number of vehicles at tick {current_time}: {len(vehicles)}')

    # Check if it's time to switch to the second map based on tick count and vehicle count
    if tick_count > 200:
        timetochange = True
    if timetochange:
        if len(vehicles) < VEHICLE_THRESHOLD:
            logger.warning("Reset detected! Switching to the second signal list.")
            current_map = 2
            current_list = second_list
            current_timer = second_original_timer.copy()  # Reset to second timer
            current_index = 0
            last_switch_tick = current_time  # Reset switch tick to current time
            timetochange = False  # Reset flag

    # Log number of vehicles and waiting vehicles in each leg
    leg_vehicle_counts = defaultdict(int)
    waiting_vehicle_counts = defaultdict(int)

    # Count vehicles and waiting vehicles (waiting if speed is 0 or near 0)
    for vehicle in vehicles:
        leg_vehicle_counts[vehicle.leg] += 1
        if vehicle.speed <= speed_threshold:  # A small threshold to consider a vehicle as waiting
            waiting_vehicle_counts[vehicle.leg] += 1

    # Store the current tick's waiting vehicle counts in history
    waiting_counts_history.append(waiting_vehicle_counts.copy())

    # Log the waiting vehicle counts and their changes
    if len(waiting_counts_history) == WINDOW_SIZE:
        previous_waiting_counts = waiting_counts_history[0]  # Get waiting counts from WINDOW_SIZE ticks ago
        change_in_waiting_counts = {leg: waiting_vehicle_counts[leg] - previous_waiting_counts.get(leg, 0) for leg in waiting_vehicle_counts}
        logger.info(f"Change in waiting vehicles for the last {WINDOW_SIZE} ticks: {change_in_waiting_counts}")

        # Apply adaptive logic continuously
        active_group_legs = current_list[current_index]
        active_changes = [change_in_waiting_counts.get(leg, 0) for leg in active_group_legs]

        # If the waiting vehicles have significantly decreased, prolong the green light
        if all(change < -4 for change in active_changes) and not adjustment_made:
            logger.info(f"Prolonging green period for {active_group_legs} by {PROLONG_TICKS} ticks.")
            current_timer[current_index] += PROLONG_TICKS  # Temporarily adjust for the current cycle
            adjustment_made = True

        # If the waiting vehicles have not significantly decreased, reduce the green light
        elif all(change > -1 for change in active_changes) and not adjustment_made:
            logger.info(f"Reducing green period for {active_group_legs} by {REDUCE_TICKS} ticks.")
            current_timer[current_index] = max(1, current_timer[current_index] - REDUCE_TICKS)  # Ensure it doesn't go below 1 tick
            adjustment_made = True

    # Get the green light duration for the current signal group based on ticks
    green_light_duration = current_timer[current_index]
    elapsed_ticks = current_time - last_switch_tick
    remaining_ticks = green_light_duration - elapsed_ticks

    # Log remaining time for current signal cycle in ticks
    logger.info(f"Remaining ticks for current signal group: {remaining_ticks} ticks")

    # Cycle through the current list based on the green light duration in ticks
    if elapsed_ticks >= green_light_duration:
        current_index = (current_index + 1) % len(current_list)  # Move to the next signal group
        logger.info(f"Switching to signal group: {current_list[current_index]}")
        last_switch_tick = current_time  # Update the last switch tick to current time

        # Reset the timer to original for the next cycle
        if current_map == 1:
            current_timer = original_timer.copy()  # Reset for first map
        else:
            current_timer = second_original_timer.copy()  # Reset for second map

        adjustment_made = False  # Allow adjustments for the next cycle

    # Log if second list is used
    if current_list == second_list:
        logger.info(f"Second list is used")
    else:
        logger.info(f"First list is used")

    # Get the active signal group
    active_group = current_list[current_index]
    logger.info(f"Active signal group: {active_group}")

    # Prepare next signals
    next_signals = []
    signal_group_vehicle_counts = defaultdict(int)

    # Count vehicles based on the leg's signal groups
    for vehicle in vehicles:
        for leg in legs:
            if vehicle.leg == leg.name:  # Assuming legs have names and signal groups
                # Count vehicles associated with signal groups in the leg
                for signal_group in leg.signal_groups:  # Access signal groups via the leg
                    signal_group_vehicle_counts[signal_group] += 1

    # Set signals to green if they are in the active group, otherwise set them to red
    for signal in signals:
        # Track the signal state based on signal.state
        if signal.state == "green":
            signal_state_durations[signal.name]["green"] += 1
        elif signal.state == "red":
            signal_state_durations[signal.name]["red"] += 1
        elif signal.state == "amber":
            signal_state_durations[signal.name]["amber"] += 1
        elif signal.state == "redamber":
            signal_state_durations[signal.name]["redamber"] += 1

        # Track how long the signal has been part of the active group
        if signal.name in active_group:
            active_group_durations[signal.name] += 1

        # Update signal state for the next tick
        if signal.name in active_group:
            next_signals.append(SignalDto(name=signal.name, state="green"))
            logger.info(f"Signal {signal.name} is green (in active group) with {signal_group_vehicle_counts[signal.name]} vehicles.")
        else:
            next_signals.append(SignalDto(name=signal.name, state="red"))
            logger.info(f"Signal {signal.name} is red (not in active group) with {signal_group_vehicle_counts[signal.name]} vehicles.")

    # Update the text file silently after each tick
    log_results_to_file(current_map)

    # Return the updated signals to the simulation
    response = TrafficSimulationPredictResponseDto(
        signals=next_signals
    )

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
