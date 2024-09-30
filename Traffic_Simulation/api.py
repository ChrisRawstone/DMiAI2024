import uvicorn
from fastapi import FastAPI
import datetime
import time
from loguru import logger
from pydantic import BaseModel
from sim.dtos import TrafficSimulationPredictResponseDto, TrafficSimulationPredictRequestDto, SignalDto
from collections import defaultdict

# Track waiting time for each signal group
pair_waiting_time = defaultdict(lambda: 0)
combination_waiting_time = defaultdict(lambda: 0)  # Initialize the missing waiting time tracker

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

# Fairness parameters
FAIRNESS_THRESHOLD = 50  # Number of ticks after which a pair must be prioritized
MIN_GREEN_DURATION = 15  # Minimum duration a signal pair must stay green (in ticks)

# Track the last green signal pair and its activation time
current_green_pair = None
current_green_start_time = 0

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
    global current_green_pair, current_green_start_time

    # Decode request
    vehicles = request.vehicles
    signals = request.signals
    signal_groups = request.signal_groups
    legs = request.legs
    allowed_green_signal_combinations = request.allowed_green_signal_combinations
    current_time = request.simulation_ticks

    logger.info(f'Number of vehicles at tick {request.simulation_ticks}: {len(vehicles)}')

    # Detect reset condition
    if len(vehicles) < 5:
        logger.warning("Reset detected! Resetting current green signal and start time.")
        current_green_pair = None
        current_green_start_time = current_time  # Reset the start time to current tick

    # Track vehicle counts per signal pair
    pair_vehicle_counts = defaultdict(int)

    # Count vehicles per leg and associate them with legal green signal pairs
    leg_vehicle_counts = defaultdict(int)
    for vehicle in vehicles:
        leg_vehicle_counts[vehicle.leg] += 1

    # Calculate vehicle counts for each valid pair
    valid_pairs = generate_unique_pairs(allowed_green_signal_combinations)
    for pair in valid_pairs:
        total_count = sum(
            leg_vehicle_counts[leg.name]
            for leg in legs
            for group in leg.signal_groups
            if group in pair
        )
        pair_vehicle_counts[pair] = total_count

    # Print vehicle counts for all pairs
    logger.info(f"Vehicle counts for each pair at tick {current_time}:")
    for pair, count in pair_vehicle_counts.items():
        logger.info(f"Pair {pair}: with vehicle count: {count}")

    # Increment waiting times for pairs that are not green
    for signal in signals:
        if signal.state != "green":
            combination_waiting_time[signal.name] += 1
        else:
            combination_waiting_time[signal.name] = 0  # Reset waiting time if it's green

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
        current_green_pair = selected_pair
        current_green_start_time = current_time  # Reset timer to current tick after selection
        logger.info(f"Selected signal pair for green: {selected_pair}")

    # Set all signals in the selected pair to green, others to red
    next_signals = []
    processed_signals = set()  # Track the signals that have been processed

    # First, make sure the selected pair is processed and both signals are set to green
    logger.info(f"Setting selected pair {selected_pair} to green")
    for group in selected_pair:
        if group not in processed_signals:
            # logger.info(f"Adding signal {group} to next_signals with state green")
            next_signals.append(SignalDto(name=group, state="green"))
            processed_signals.add(group)

    # Now, process the rest of the pairs and set remaining signals to red
    for pair in valid_pairs:
        if pair != selected_pair:
            for group in pair:
                if group not in processed_signals:
                    # logger.info(f"Adding signal {group} to next_signals with state red")
                    next_signals.append(SignalDto(name=group, state="red"))
                    processed_signals.add(group)

    #sort next_signals by name
    next_signals = sorted(next_signals, key=lambda x: x.name)

    # Print the current status of signals and their associated vehicle counts with color coding
    for signal in next_signals:
        # Find the pair the signal belongs to
        relevant_pair = None
        for pair in valid_pairs:
            if signal.name in pair:
                relevant_pair = pair
                break

        if relevant_pair:
            # Get the vehicle count for the relevant pair
            vehicle_count = pair_vehicle_counts.get(relevant_pair, 0)
        else:
            vehicle_count = 0

        color = "green" if signal.state == "green" else "red"

        # Set the corresponding color for the log output
        color_code = "\033[92m" if signal.state == "green" else "\033[91m"  # Green for 'green', Red for 'red'
        reset_color = "\033[0m"  # Reset color code to default

        # Log with the appropriate color and vehicle count for the pair
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
