import uvicorn
from fastapi import FastAPI
import datetime
import time
from loguru import logger
from pydantic import BaseModel
from sim.dtos import TrafficSimulationPredictResponseDto, TrafficSimulationPredictRequestDto, SignalDto
from collections import defaultdict

# Track waiting time for each signal group
group_waiting_time = defaultdict(lambda: 0)
group_last_activation = defaultdict(lambda: 0)

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

# Track waiting time for each legal green signal combination
combination_waiting_time = defaultdict(lambda: 0)

# Fairness parameters
FAIRNESS_THRESHOLD = 20  # Number of ticks after which a group must be prioritized
MIN_GREEN_TIME = 6  # Minimum time a signal must remain green
MAX_WAITING_TIME = 60  # Max waiting time before a group is forced green

@app.post('/predict', response_model=TrafficSimulationPredictResponseDto)
def predict_endpoint(request: TrafficSimulationPredictRequestDto):
    # Decode request
    vehicles = request.vehicles
    signals = request.signals
    signal_groups = request.signal_groups
    legs = request.legs
    allowed_green_signal_combinations = request.allowed_green_signal_combinations

    logger.info(f'Number of vehicles at tick {request.simulation_ticks}: {len(vehicles)}')

    # Track vehicle counts per signal combination
    combination_vehicle_counts = defaultdict(int)

    # Count vehicles per leg and associate them with legal green signal combinations
    leg_vehicle_counts = defaultdict(int)
    for vehicle in vehicles:
        leg_vehicle_counts[vehicle.leg] += 1
        logger.info(f"Vehicle {vehicle} assigned to leg {vehicle.leg}, count now {leg_vehicle_counts[vehicle.leg]}")

    for combination in allowed_green_signal_combinations:
        total_count = sum(leg_vehicle_counts[leg.name] for leg in legs if leg.signal_groups[0] in combination.groups)
        combination_vehicle_counts[combination.name] = total_count
        logger.info(f"Total count for combination {combination.name}: {total_count}")

    # Increment waiting times for combinations that are not green
    for signal in signals:
        if signal.state != "green":
            combination_waiting_time[signal.name] += 1
        else:
            combination_waiting_time[signal.name] = 0  # Reset waiting time if it's green

    # Sort combinations by vehicle count, then by waiting time
    sorted_combinations = sorted(
        allowed_green_signal_combinations,
        key=lambda comb: (-combination_vehicle_counts[comb.name], -combination_waiting_time[comb.name])
    )

    # Select the legal green signal combination with the most cars or the one that has waited too long
    selected_combination = None
    for combination in sorted_combinations:
        if combination_waiting_time[combination.name] >= FAIRNESS_THRESHOLD:
            selected_combination = combination.name  # Prioritize based on fairness
            break
        elif combination_vehicle_counts[combination.name] > 0:
            selected_combination = combination.name  # Prioritize based on vehicle count
            break

    # If no combination is selected, prioritize by max waiting time
    if not selected_combination:
        for combination in allowed_green_signal_combinations:
            if combination_waiting_time[combination.name] >= MAX_WAITING_TIME:
                selected_combination = combination.name
                break

    # Default to the first combination if no selection is made (shouldn't happen)
    if not selected_combination:
        logger.warning("No combination selected, defaulting to the first combination")
        selected_combination = allowed_green_signal_combinations[0].name

    logger.info(f"Selected signal combination for green: {selected_combination}")

    # Print the selected group for debugging
    print(f"Selected signal group for green: {selected_combination}")

    # Set all signals in the selected combination to green, others to red
    next_signals = [
        SignalDto(name=signal.name, state="green" if signal.name == selected_combination else "red")
        for signal in signals
    ]

    # Return the next signals to the evaluation service
    response = TrafficSimulationPredictResponseDto(
        signals=next_signals
    )

    return response   

if __name__ == '__main__':

    uvicorn.run(
        'api:app',
        host=HOST,
        port=PORT
    )
