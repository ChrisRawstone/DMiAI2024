import uvicorn
from fastapi import FastAPI
import datetime
from loguru import logger
from pydantic import BaseModel
from sim.dtos import TrafficSimulationPredictResponseDto, TrafficSimulationPredictRequestDto, SignalDto
from collections import defaultdict

"""
Max values for vehicle count data at tick 250:
Vehicle count data at tick 250 map 1:
Leg A1: 38 vehicles
Leg A2: 22 vehicles
Leg B1: 18 vehicles
Leg B2: 51 vehicles

Vehicle count data at tick 250 for map 2:
Leg A1: 24 vehicles
Leg A2: 17 vehicles
Leg B1: 16 vehicles
Leg B2: 14 vehicles
"""

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
first_timer = [29, 20, 20, 35]  # Green light durations for first map in ticks

second_list = [
    ['A1', 'A1RightTurn', 'A1LeftTurn'],
    ['A2', 'A2RightTurn', 'A2LeftTurn'],
    ['B1', 'B1RightTurn', 'A1RightTurn'],
    ['B2', 'A2RightTurn', 'B1RightTurn']  # Optimize this, B1 and B1RightTurn seem redundant here
]
second_timer = [28, 27, 27, 25]  # Green light durations for second map in ticks

# Initial setup
app = FastAPI()
current_list = first_list
current_timer = first_timer  # Start with first timer
current_index = 0
last_switch_tick = 0  # Use ticks to track when the last switch happened
tick_count = 0

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
    global current_list, current_timer, current_index, tick_count, last_switch_tick, timetochange

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
            current_list = second_list
            current_timer = second_timer  # Switch to second timer
            current_index = 0
            last_switch_tick = current_time  # Reset switch tick to current time
            timetochange = False  # Reset flag

    # Get the green light duration for the current signal group based on ticks
    green_light_duration = current_timer[current_index]

    # Calculate elapsed ticks since the last signal switch
    elapsed_ticks = current_time - last_switch_tick
    remaining_ticks = green_light_duration - elapsed_ticks

    # Log remaining time for current signal cycle in ticks
    logger.info(f"Remaining ticks for current signal group: {remaining_ticks} ticks")

    # Cycle through the current list based on the green light duration in ticks
    if elapsed_ticks >= green_light_duration:
        current_index = (current_index + 1) % len(current_list)  # Move to the next signal group
        logger.info(f"Switching to signal group: {current_list[current_index]}")
        last_switch_tick = current_time  # Update the last switch tick to current time

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
        if signal.name in active_group:
            next_signals.append(SignalDto(name=signal.name, state="green"))
            logger.info(f"\033[92mSignal {signal.name} is {signal.state} with {signal_group_vehicle_counts[signal.name]} vehicles.\033[0m")
        else:
            next_signals.append(SignalDto(name=signal.name, state="red"))
            logger.info(f"\033[91mSignal {signal.name} is {signal.state} with {signal_group_vehicle_counts[signal.name]} vehicles.\033[0m")

    # Return the updated signals to the simulation
    response = TrafficSimulationPredictResponseDto(
        signals=next_signals
    )

    return response


if __name__ == '__main__':
    uvicorn.run(
        'api:app',
        host="0.0.0.0",
        port=8080
    )
