import uvicorn
from fastapi import FastAPI
import datetime
import time
from loguru import logger
from pydantic import BaseModel
from sim.dtos import TrafficSimulationPredictResponseDto, TrafficSimulationPredictRequestDto, SignalDto
from collections import defaultdict

# Time-based cycling parameters
CYCLE_DURATION = 20  # Seconds to change signal groups
VEHICLE_THRESHOLD = 5  # Vehicle count to trigger reset
RESET_TICK_THRESHOLD = 50  # Check reset only after 50 ticks
OVERLAP_DURATION = 1  # Seconds of overlap between two signal groups

# Lists for signal combinations
# first_list = [
#     ['A1', 'A2'],
#     ['A1', 'A1LeftTurn'],
#     ['A2', 'A2LeftTurn'],
#     ['B1', 'B2'],
#     ['B1', 'B1LeftTurn'],
#     ['B2', 'B2LeftTurn']
# ]

# second_list = [
#     ['A1', 'A2', 'A2RightTurn', 'A1RightTurn'],
#     ['A2', 'A2RightTurn', 'A2LeftTurn'],
#     ['B1', 'B2', 'B1RightTurn'],
#     ['B2', 'A2RightTurn', 'B1RightTurn'],
#     ['B1', 'B1RightTurn', 'A1RightTurn'],
#     ['A1', 'A1RightTurn', 'A1LeftTurn'],
#     ['A1LeftTurn', 'A2LeftTurn', 'B1RightTurn']
# ]

first_list = [
    ['A1', 'A1LeftTurn'],
    ['A2', 'A2LeftTurn'],
    ['B1', 'B1LeftTurn'],
    ['B2', 'B2LeftTurn']
]

second_list = [
    ['A1', 'A1RightTurn', 'A1LeftTurn'],
    ['A2', 'A2RightTurn', 'A2LeftTurn'],
    ['B1', 'B1RightTurn', 'A1RightTurn'],
    ['B1', 'B2', 'B1RightTurn'],
]
    

# Initial setup
app = FastAPI()
current_list = first_list
current_index = 0
start_time = time.time()
tick_count = 0
previous_group = None  # To track the previous signal group for overlap
overlap_start_time = None  # To track the start time of overlap

@app.get('/api')
def hello():
    return {
        "service": "traffic-simulation-usecase",
        "uptime": '{}'.format(datetime.timedelta(seconds=time.time() - start_time))
    }

@app.get('/')
def index():
    return "Your endpoint is running!"

timetochange = False
@app.post('/predict', response_model=TrafficSimulationPredictResponseDto)
def predict_endpoint(request: TrafficSimulationPredictRequestDto):
    global current_list, current_index, tick_count, start_time, timetochange, previous_group, overlap_start_time
    # Decode request data
    vehicles = request.vehicles
    signals = request.signals
    legs = request.legs  # Assuming legs contain the signal groups
    current_time = request.simulation_ticks
    tick_count = current_time
    logger.info(f'Number of vehicles at tick {current_time}: {len(vehicles)}')

    if tick_count > 200:
        timetochange = True
    if timetochange:
        if len(vehicles) < 5:
            current_list = second_list

    # Detect reset condition
    if len(vehicles) < VEHICLE_THRESHOLD and tick_count >= RESET_TICK_THRESHOLD:
        logger.warning("Reset detected! Switching to the second signal list.")
        current_list = second_list
        current_index = 0

    # Cycle through the current list every 15 seconds
    if time.time() - start_time >= CYCLE_DURATION:
        previous_group = current_list[current_index]  # Store the current group as the previous group
        current_index = (current_index + 1) % len(current_list)
        logger.info(f"Switching to signal group: {current_list[current_index]}")
        overlap_start_time = time.time()  # Start overlap period
        start_time = time.time()

    # Log if second list is used
    if current_list == second_list:
        logger.info(f"Second list is used")
    else:
        logger.info(f"First list is used")
        
    # Get the active signal group
    active_group = current_list[current_index]
    logger.info(f"Active signal group: {active_group}")

    # Prepare next signals with overlap handling
    next_signals = []
    signal_group_vehicle_counts = defaultdict(int)

    # Count vehicles based on the leg's signal groups
    for vehicle in vehicles:
        for leg in legs:
            if vehicle.leg == leg.name:  # Assuming legs have names and signal groups
                # Count vehicles associated with signal groups in the leg
                for signal_group in leg.signal_groups:  # Access signal groups via the leg
                    signal_group_vehicle_counts[signal_group] += 1

    # Set signals to green if they are in the active group or in the previous group during overlap
    for signal in signals:
        if signal.name in active_group:
            next_signals.append(SignalDto(name=signal.name, state="green"))
            logger.info(f"\033[92mSignal {signal.name} is green with {signal_group_vehicle_counts[signal.name]} vehicles.\033[0m")
        elif previous_group and signal.name in previous_group and overlap_start_time and (time.time() - overlap_start_time < OVERLAP_DURATION):
            # Keep previous group signals green for the overlap duration
            next_signals.append(SignalDto(name=signal.name, state="green"))
            logger.info(f"\033[93mSignal {signal.name} is still green (overlap) with {signal_group_vehicle_counts[signal.name]} vehicles.\033[0m")
        else:
            next_signals.append(SignalDto(name=signal.name, state="red"))
            logger.info(f"\033[91mSignal {signal.name} is red with {signal_group_vehicle_counts[signal.name]} vehicles.\033[0m")

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
