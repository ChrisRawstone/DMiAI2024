from multiprocessing import Process, Queue
from time import sleep, time
from environment import load_and_run_simulation
from collections import defaultdict
from dtos import SignalDto
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Initialize queues
input_queue = Queue()
output_queue = Queue()
error_queue = Queue()


# Track signal durations and vehicle counts
green_signal_durations = defaultdict(int)
leg_vehicle_counts_total = defaultdict(int)

# Track the last green signal pair and its activation time
current_green_pair = None
current_green_start_time = 0
previous_green_pair = None  # To track the previous signal pair for overlap
previous_green_start_time = 0
pair_waiting_time = defaultdict(lambda: 0)


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


# Constants at the top of the script
FAIRNESS_THRESHOLD = 50  # number of ticks after which a pair must be prioritized
MIN_GREEN_DURATION = 15   # Minimum duration a signal pair must stay green (in ticks)
OVERLAP_DURATION = 1    # Duration where both old and new signals remain green (in ticks)
UNFAIR_GREEN_DURATION = 20  # Green duration when a pair is chosen due to unfairness

def run_game():
    test_duration_seconds = 600
    random = True
    configuration_file = "models/1/glue_configuration.yaml"
    start_time = time()

    # Start simulation process
    p = Process(target=load_and_run_simulation, args=(configuration_file,
                                                      start_time,
                                                      test_duration_seconds,
                                                      random,
                                                      input_queue,
                                                      output_queue,
                                                      error_queue))
    p.start()

    # Wait for the simulation to initialize
    sleep(0.2)

    actions = {}
    global current_green_pair, current_green_start_time, previous_green_pair, previous_green_start_time
    global current_green_duration  # This needs to be global to persist between runs

    # Initialize a new dictionary to track waiting times for pairs
    pair_waiting_time = defaultdict(int)
    current_green_duration = MIN_GREEN_DURATION  # Set initial green duration

    while True:
        # Get the state from the output queue
        state = output_queue.get()

        if state.is_terminated:
            p.join()
            break

        # Reset the green signal pair if simulation reset is detected
        if len(state.vehicles) < 5:
            print("Reset detected! Resetting current green signal and start time.")
            current_green_pair = None
            current_green_start_time = state.simulation_ticks
            current_green_duration = MIN_GREEN_DURATION  # Reset to default after a reset

        # Track vehicle counts per signal pair
        pair_vehicle_counts = defaultdict(int)

        # Count vehicles per leg and associate them with legal green signal pairs
        leg_vehicle_counts = defaultdict(int)
        for vehicle in state.vehicles:
            leg_vehicle_counts[vehicle.leg] += 1

        # Calculate vehicle counts for each valid pair
        valid_pairs = generate_unique_pairs(state.allowed_green_signal_combinations)
        for pair in valid_pairs:
            total_vehicle_count = 0
            for leg in state.legs:
                if any(group in pair for group in leg.signal_groups):
                    total_vehicle_count += leg_vehicle_counts[leg.name]
            pair_vehicle_counts[pair] = total_vehicle_count

        # Track waiting times for pairs that are not green
        for pair in valid_pairs:
            # Check if the pair is currently green
            pair_is_green = any(
                signal.state == "green" and signal.name in pair for signal in state.signals
            )

            # If the pair is not green, increment its waiting time
            if not pair_is_green:
                pair_waiting_time[pair] += 1
            else:
                # Reset waiting time for the pair if it's green
                pair_waiting_time[pair] = 0

        # Track how long each signal is green
        for signal in state.signals:
            if signal.state == "green":
                green_signal_durations[signal.name] += 1  # Increment green duration

        # Check if fairness should be triggered, and set the green duration accordingly
        if current_green_pair and (state.simulation_ticks - current_green_start_time) < current_green_duration:
            selected_pair = current_green_pair
        else:
            # Sort pairs by vehicle count, then by waiting time
            sorted_pairs = sorted(
                valid_pairs,
                key=lambda pair: (-pair_vehicle_counts[pair], -pair_waiting_time[pair])
            )

            # Select the legal green signal pair with the most cars or the one that has waited too long
            selected_pair = None
            fairness_triggered = False  # Flag to track if fairness was triggered

            for pair in sorted_pairs:
                if pair_waiting_time[pair] >= FAIRNESS_THRESHOLD:
                    print(f"Pair {pair} has waited too long, selecting it due to fairness.")
                    selected_pair = pair  # Prioritize based on fairness
                    fairness_triggered = True  # Mark that fairness was triggered
                    break
                elif pair_vehicle_counts[pair] > 0:
                    selected_pair = pair  # Prioritize based on vehicle count
                    break

            # If no pair is selected, prioritize by max waiting time
            if not selected_pair:
                for pair in valid_pairs:
                    if pair_waiting_time[pair] >= FAIRNESS_THRESHOLD:
                        selected_pair = pair
                        fairness_triggered = True  # Mark that fairness was triggered
                        print(f"Pair {pair} has waited too long, selecting it due to fairness.")
                        break

            # Default to the first pair if no selection is made
            if not selected_pair:
                selected_pair = valid_pairs[0]

            # Set the new green pair and its start time
            previous_green_pair = current_green_pair
            previous_green_start_time = current_green_start_time
            current_green_pair = selected_pair
            current_green_start_time = state.simulation_ticks

            # Use unfair duration if fairness was triggered, otherwise use default
            current_green_duration = UNFAIR_GREEN_DURATION if fairness_triggered else MIN_GREEN_DURATION

        # Set all signals in the selected pair to green, others to red
        next_signals = []
        processed_signals = set()  # Track the signals that have been processed

        # Apply overlap logic: if the previous pair exists and is still within the overlap period, keep them green
        if previous_green_pair and (state.simulation_ticks - previous_green_start_time) < (MIN_GREEN_DURATION + OVERLAP_DURATION):
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

        # Track the vehicle counts
        for leg, count in leg_vehicle_counts.items():
            leg_vehicle_counts_total[leg] += count

        # Send the next signals to the simulation input queue
        input_queue.put({signal.name: signal.state for signal in next_signals})

    # End of simulation, generate plots
    generate_plots()


import os

def generate_plots():
    # Create output directory if it doesn't exist
    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert data to DataFrame for seaborn
    green_signal_df = pd.DataFrame(list(green_signal_durations.items()), columns=["Signal", "Green Duration"])
    leg_vehicle_df = pd.DataFrame(list(leg_vehicle_counts_total.items()), columns=["Leg", "Total Vehicles"])

    # Set seaborn style
    sns.set(style="whitegrid")

    # Plot 1: Green signal duration
    plt.figure(figsize=(10, 6))
    # Assign 'Signal' to hue and set legend to False
    sns.barplot(x="Signal", y="Green Duration", data=green_signal_df, hue="Signal", palette="Blues_d", legend=False)   
    plt.title("Total Duration Each Signal Stayed Green", fontsize=16)
    plt.xlabel("Signal", fontsize=14)
    plt.ylabel("Green Duration (Ticks)", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join(output_dir, "green_signal_durations.png"))
    plt.close()

    # Plot 2: Total vehicle count per leg
    plt.figure(figsize=(10, 6))
        
    # Assign 'Leg' to hue and set legend to False
    sns.barplot(x="Leg", y="Total Vehicles", data=leg_vehicle_df, hue="Leg", palette="Greens_d", legend=False)
    plt.title("Total Vehicles from Each Leg", fontsize=16)
    plt.xlabel("Leg", fontsize=14)
    plt.ylabel("Vehicle Count", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    
    
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, "leg_vehicle_counts.png"))
    plt.close()
    





if __name__ == '__main__':
    run_game()
