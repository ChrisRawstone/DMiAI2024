import uvicorn
from fastapi import FastAPI
import datetime
import time
from loguru import logger
from pydantic import BaseModel
from sim.dtos import TrafficSimulationPredictResponseDto, TrafficSimulationPredictRequestDto, SignalDto
from collections import defaultdict
#To do: 
# 1. Fix counter
# 2. Fix the logic for the signal pairs, sometimes 3 signals can be green at once. 
#allowed_green_signal_combinations: [AllowedGreenSignalCombinationDto(name='A1', groups=['A1LeftTurn', 'A2', 'A1RightTurn']), AllowedGreenSignalCombinationDto(name='A1LeftTurn', groups=['A1', 'A2LeftTurn']), AllowedGreenSignalCombinationDto(name='A1RightTurn', groups=['A1', 'A2RightTurn', 'B1RightTurn']), AllowedGreenSignalCombinationDto(name='A2', groups=['A2LeftTurn', 'A2RightTurn', 'A1']), AllowedGreenSignalCombinationDto(name='A2RightTurn', groups=['A2', 'A2LeftTurn', 'B2', 'A1RightTurn', 'B1RightTurn']), AllowedGreenSignalCombinationDto(name='B1', groups=['B1RightTurn', 'B2']), AllowedGreenSignalCombinationDto(name='B1RightTurn', groups=['B1', 'A2RightTurn', 'A1RightTurn']), AllowedGreenSignalCombinationDto(name='B2', groups=['B1'])]

# Fairness parameters
FAIRNESS_THRESHOLD = 40  # Number of ticks after which a pair must be prioritized
MIN_GREEN_DURATION = 10   # Minimum duration a signal pair must stay green (in ticks)
OVERLAP_DURATION = 2      # Duration where both old and new signals remain green (in ticks)

# Track the last green signal pair and its activation time
current_green_pair = None
current_green_start_time = 0
previous_green_pair = None  # To track the previous signal pair for overlap
previous_green_start_time = 0
pair_waiting_time = defaultdict(lambda: 0)

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

def generate_unique_pairs(allowed_green_signal_combinations):
    """
    Generate valid unique pairs from allowed_green_signal_combinations.
    Ensures no duplicate pairs like (A1, A2) and (A2, A1).
    """
    valid_pairs = set()

    for combination in allowed_green_signal_combinations:
        for group in combination.groups:
            if combination.name != group:
                # Sort the pair to ensure uniqueness (A1, A2) == (A2, A1)
                pair = tuple(sorted([combination.name, group]))
                valid_pairs.add(pair)

    return list(valid_pairs)

@app.post('/predict', response_model=TrafficSimulationPredictResponseDto)
def predict_endpoint(request: TrafficSimulationPredictRequestDto):
    global current_green_pair, current_green_start_time, previous_green_pair, previous_green_start_time

    # Decode request
    vehicles = request.vehicles
    signals = request.signals
    legs = request.legs
    allowed_green_signal_combinations = request.allowed_green_signal_combinations
    current_time = request.simulation_ticks
    print(f"allowed_green_signal_combinations: {allowed_green_signal_combinations}")
    logger.info(f'Number of vehicles at tick {request.simulation_ticks}: {len(vehicles)}')

    # Detect reset condition
    if len(vehicles) < 5:
        logger.warning("Reset detected! Resetting current green signal and start time.")
        current_green_pair = None
        current_green_start_time = current_time  # Reset the start time to current tick

    # Track vehicle counts per signal group
    signal_group_vehicle_counts = defaultdict(int)

    # Count vehicles for each leg and signal group
    for vehicle in vehicles:
        for leg in legs:
            if vehicle.leg == leg.name:
                # For each leg, increment the vehicle count for each signal group it is associated with
                for signal_group in leg.signal_groups:
                    signal_group_vehicle_counts[signal_group] += 1

    # Now associate the signal groups with their pairs
    valid_pairs = generate_unique_pairs(allowed_green_signal_combinations)
    pair_vehicle_counts = defaultdict(int)

    for pair in valid_pairs:
        total_vehicle_count = 0
        # Sum the vehicle counts for each signal group in the pair
        for group in pair:
            total_vehicle_count += signal_group_vehicle_counts[group]
        pair_vehicle_counts[pair] = total_vehicle_count

    # Print vehicle counts for all pairs
    logger.info(f"Vehicle counts for each pair at tick {current_time}:")
    for pair, count in pair_vehicle_counts.items():
        logger.info(f"Pair {pair}: with vehicle count: {count}")

    # Increment waiting times for pairs that are not green
    for signal in signals:
        if signal.state != "green":
            pair_waiting_time[signal.name] += 1
        else:
            pair_waiting_time[signal.name] = 0  # Reset waiting time if it's green

    # Keep the current green pair active for the minimum green duration
    if current_green_pair and (current_time - current_green_start_time) < MIN_GREEN_DURATION:
        selected_pair = current_green_pair
        logger.info(f"Keeping current green signal pair: {selected_pair} (active for {current_time - current_green_start_time} ticks)")
    else:
        # Sort pairs by vehicle count, then by waiting time
        sorted_pairs = sorted(
            valid_pairs,
            key=lambda pair: (-pair_vehicle_counts[pair], -pair_waiting_time[pair])
        )

        # Select the legal green signal pair with the most cars or the one that has waited too long
        selected_pair = None
        for pair in sorted_pairs:
            if pair_waiting_time[pair] >= FAIRNESS_THRESHOLD:
                selected_pair = pair  # Prioritize based on fairness
                break
            elif pair_vehicle_counts[pair] > 0:
                selected_pair = pair  # Prioritize based on vehicle count
                break

        # If no pair is selected, prioritize by max waiting time
        if not selected_pair:
            for pair in valid_pairs:
                if pair_waiting_time[pair] >= FAIRNESS_THRESHOLD:
                    selected_pair = pair
                    break

        # Default to the first pair if no selection is made (shouldn't happen)
        if not selected_pair:
            logger.warning("No pair selected, defaulting to the first pair")
            selected_pair = valid_pairs[0]

        # Set the new green pair and its start time
        previous_green_pair = current_green_pair  # Track the previous pair for overlap
        previous_green_start_time = current_green_start_time
        current_green_pair = selected_pair
        current_green_start_time = current_time

        logger.info(f"Selected signal pair for green: {selected_pair}")

    # Set all signals in the selected pair to green, others to red
    next_signals = []
    processed_signals = set()  # Track the signals that have been processed

    # Apply overlap logic: if the previous pair exists and is still within the overlap period, keep them green
    if previous_green_pair and (current_time - previous_green_start_time) < (MIN_GREEN_DURATION + OVERLAP_DURATION):
        logger.info(f"Keeping previous pair {previous_green_pair} green for overlap.")
        for group in previous_green_pair:
            if group not in processed_signals:
                next_signals.append(SignalDto(name=group, state="green"))
                processed_signals.add(group)

    # Set the selected pair to green
    for group in selected_pair:
        if group not in processed_signals:
            next_signals.append(SignalDto(name=group, state="green"))
            processed_signals.add(group)

    # Set the rest to red
    for pair in valid_pairs:
        if pair != selected_pair:
            for group in pair:
                if group not in processed_signals:
                    next_signals.append(SignalDto(name=group, state="red"))
                    processed_signals.add(group)

    # Sort next_signals by name for consistency
    next_signals = sorted(next_signals, key=lambda x: x.name)

    # Print the current status of signals and their associated vehicle counts with color coding
    for signal in next_signals:
        relevant_pair = None
        for pair in valid_pairs:
            if signal.name in pair:
                relevant_pair = pair
                break

        vehicle_count = signal_group_vehicle_counts.get(signal.name, 0)
        color = "green" if signal.state == "green" else "red"
        color_code = "\033[92m" if signal.state == "green" else "\033[91m"
        reset_color = "\033[0m"
        logger.info(f"{color_code}Signal {signal.name} is {color} with {vehicle_count} vehicles.{reset_color}")

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
