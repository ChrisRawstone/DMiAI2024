import pickle
from multiprocessing import Process, Queue
from time import sleep, time
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List

from environment_gui import load_and_run_simulation  # Ensure this is correctly imported based on your project structure

import logging
import os
import argparse

# --------------------- Configuration ---------------------

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# --------------------- Data Classes ---------------------

@dataclass
class EnhancedState:
    vehicle_counts: List[int]
    average_speeds: List[int]
    waiting_times: List[int]
    current_signal_states: List[str]
    elapsed_time_since_last_action: int

# --------------------- RL Agent Class ---------------------

class RLAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        Initialize the RL Agent with given parameters.
        """
        self.q_table = defaultdict(float)  # Mapping from state-action to value
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def get_state_key(self, state: EnhancedState):
        """
        Convert the enhanced state to a tuple to use as a key in the Q-table.
        Discretizes average speeds and waiting times into bins.
        """
        # Discretize average speeds into bins
        speed_bins = np.digitize(state.average_speeds, bins=[0, 5, 10, 15, 20, 25, 30])
        # Discretize waiting times into bins
        waiting_time_bins = np.digitize(state.waiting_times, bins=[0, 10, 20, 30, 40, 50, 60])

        return (
            tuple(state.vehicle_counts),
            tuple(speed_bins),
            tuple(waiting_time_bins),
            tuple(state.current_signal_states),
            state.elapsed_time_since_last_action
        )

    def choose_action(self, state: EnhancedState):
        """
        Choose an action based on the current state using an epsilon-greedy policy.
        """
        if np.random.rand() <= self.epsilon:
            # Explore: choose a random action
            action = np.random.choice(self.action_size)
            logging.debug(f"Exploring: Chose random action {action}")
            return action
        else:
            # Exploit: choose the best action from Q-table
            state_key = self.get_state_key(state)
            q_values = [self.q_table[(state_key, a)] for a in range(self.action_size)]
            max_q = max(q_values)
            # In case multiple actions have the same max Q-value, choose randomly among them
            actions_with_max_q = [a for a in range(self.action_size) if q_values[a] == max_q]
            action = np.random.choice(actions_with_max_q)
            logging.debug(f"Exploiting: Chose best action {action} with Q-value {max_q}")
            return action

    # Note: No update_q_value or decay_epsilon methods since this script is for evaluation only

# --------------------- Helper Functions ---------------------

def generate_allowed_configurations(allowed_green_signal_combinations):
    """
    Generate predefined traffic light phases with minimum durations to prevent rapid switching.
    """
    configurations = [
        {'signals': ['A1', 'A2'], 'min_duration': 30},  # Phase 0
        {'signals': ['B1', 'B2'], 'min_duration': 30},  # Phase 1
        {'signals': ['A1LeftTurn', 'A2LeftTurn'], 'min_duration': 20},  # Phase 2
        {'signals': ['B1LeftTurn', 'B2LeftTurn'], 'min_duration': 20},  # Phase 3
    ]
    return configurations

def get_enhanced_state_representation(state, signal_names, last_action_time, simulation_ticks):
    """
    Generate an enhanced state representation with additional features.
    """
    vehicle_counts = []
    average_speeds = []
    waiting_times = []

    for signal_name in signal_names:
        # Filter vehicles in the current lane
        vehicles_in_lane = [v for v in state.vehicles if v.leg == signal_name]
        # Identify stopped vehicles (speed < 0.5)
        stopped_vehicles = [v for v in vehicles_in_lane if v.speed < 0.5]
        vehicle_counts.append(len(stopped_vehicles))

        if vehicles_in_lane:
            # Calculate average speed in the lane
            avg_speed = sum(v.speed for v in vehicles_in_lane) / len(vehicles_in_lane)
            average_speeds.append(int(avg_speed))

            # Calculate average waiting time
            waiting_time_values = []
            for v in stopped_vehicles:
                if v.speed > 0:
                    try:
                        waiting_time = simulation_ticks - (v.distance_to_stop / v.speed)
                    except ZeroDivisionError:
                        # Safeguard, should not occur
                        waiting_time = simulation_ticks
                        logging.debug(f"ZeroDivisionError for vehicle {v}. Assigned waiting_time = {waiting_time}")
                else:
                    # Assign a high waiting time for completely stopped vehicles
                    waiting_time = simulation_ticks  # Alternatively, use a predefined max waiting time
                    logging.debug(f"Vehicle {v} is completely stopped. Assigned waiting_time = {waiting_time}")
                waiting_time_values.append(waiting_time)

            avg_waiting_time = sum(waiting_time_values) / len(stopped_vehicles) if stopped_vehicles else 0
            waiting_times.append(int(avg_waiting_time))
        else:
            average_speeds.append(0)
            waiting_times.append(0)

    # Get current signal states
    current_signal_states = [signal.state for signal in state.signals]
    # Calculate elapsed time since last action
    elapsed_time = simulation_ticks - last_action_time

    return EnhancedState(
        vehicle_counts=vehicle_counts,
        average_speeds=average_speeds,
        waiting_times=waiting_times,
        current_signal_states=current_signal_states,
        elapsed_time_since_last_action=elapsed_time
    )

def run_simulation(configuration_file, test_duration_seconds=600, random=True):
    """
    Initialize and start the traffic simulation process.
    """
    start_time = time()

    input_queue = Queue()
    output_queue = Queue()
    error_queue = Queue()
    errors = []

    p = Process(target=load_and_run_simulation, args=(
        configuration_file,
        start_time,
        test_duration_seconds,
        random,
        input_queue,
        output_queue,
        error_queue
    ))

    p.start()

    # Wait briefly to ensure the simulation starts
    sleep(0.2)

    # For logging purposes
    actions = {}

    return p, input_queue, output_queue, error_queue, actions

def run_episode(agent, configuration_file="models/1/glue_configuration.yaml",
               test_duration_seconds=600, random=True):
    """
    Run a single episode using the loaded RL agent.
    """
    p, input_queue, output_queue, error_queue, actions = run_simulation(configuration_file, test_duration_seconds, random)

    agent_initialized = False
    previous_state = None
    previous_action = None
    signal_names = []
    allowed_configurations = []
    last_action_time = 0  # Track the last time an action was taken
    previous_total_score = 0

    while True:
        try:
            state = output_queue.get(timeout=5)  # Timeout to prevent hanging
        except:
            logging.warning("No state received. Terminating simulation.")
            p.terminate()
            p.join()
            break

        if state.is_terminated:
            p.join()
            break

        simulation_ticks = state.simulation_ticks

        if not agent_initialized:
            # Initialize the agent
            signal_names = [signal.name for signal in state.signals]
            allowed_configurations = generate_allowed_configurations(state.allowed_green_signal_combinations)
            state_size = len(signal_names) * 3 + len(state.signals) + 1  # Example size calculation
            action_size = len(allowed_configurations)
            agent.state_size = state_size
            agent.action_size = action_size
            agent_initialized = True
            logging.info("RL Agent initialized.")

        # Generate the enhanced state
        current_state = get_enhanced_state_representation(state, signal_names, last_action_time, simulation_ticks)

        # Enforce minimum duration between actions (e.g., 30 ticks)
        MIN_DURATION = 30
        if current_state.elapsed_time_since_last_action < MIN_DURATION:
            # Continue with the previous action to respect the minimum duration
            action_index = previous_action if previous_action is not None else np.random.choice(agent.action_size)
            logging.debug(f"Continuing with previous action {action_index} due to minimum duration constraint.")
        else:
            # Choose a new action based on the Q-table
            action_index = agent.choose_action(current_state)
            last_action_time = simulation_ticks  # Update the last action time

        selected_configuration = allowed_configurations[action_index]['signals']
        min_duration = allowed_configurations[action_index]['min_duration']
        logging.debug(
            f"Tick {simulation_ticks}: Selected action {action_index} with configuration "
            f"{selected_configuration} and min_duration {min_duration}"
        )

        # Prepare the traffic light signals based on the selected configuration
        prediction = {"signals": []}
        for signal_name in signal_names:
            if signal_name in selected_configuration:
                prediction["signals"].append({"name": signal_name, "state": "green"})
            else:
                prediction["signals"].append({"name": signal_name, "state": "red"})

        # Send the desired signal states to the simulation
        next_signals = {signal['name']: signal['state'] for signal in prediction['signals']}
        input_queue.put(next_signals)

        # In evaluation mode, do not update Q-values or decay epsilon
        # Thus, we skip any update-related operations

        # Update tracking variables
        previous_state = current_state
        previous_action = action_index
        previous_total_score = state.total_score

    # End of simulation, return the score
    if state.total_score == 0:
        state.total_score = 1e9  # Assign a very high score if it's zero to indicate a perfect scenario

    logging.info(f"Final inverted score: {state.total_score}")

    return state.total_score

# --------------------- Main Function ---------------------

def main():
    parser = argparse.ArgumentParser(description="Run one episode using a saved RL model for traffic simulation.")

    parser.add_argument('--configuration_file', type=str, default="models/1/glue_configuration.yaml",
                        help="Path to the traffic simulation configuration file")
    parser.add_argument('--test_duration', type=int, default=300,
                        help="Duration per simulation in seconds")
    parser.add_argument('--random', action='store_true',
                        help="Run simulation in random mode")
    args = parser.parse_args()


    q_table = "improved_models/q_table_worker_9_episode_135_score_1640.0_1727806744.pkl"
    agent_params = "improved_models/agent_parameters_worker_9_episode_135_score_1640.0_1727806744.pkl"

    # Validate file paths
    if not os.path.isfile(q_table):
        logging.error(f"Q-table file '{q_table}' does not exist.")
        return
    if not os.path.isfile(agent_params):
        logging.error(f"Agent parameters file '{agent_params}' does not exist.")
        return

    # Load the saved Q-table
    with open(q_table, 'rb') as f:
        loaded_q_table = pickle.load(f)
    logging.info(f"Loaded Q-table from '{q_table}'.")

    # Load the saved agent parameters
    with open(agent_params, 'rb') as f:
        loaded_agent_params = pickle.load(f)
    logging.info(f"Loaded agent parameters from '{agent_params}'.")

    # Initialize the RL agent with loaded parameters
    agent = RLAgent(
        state_size=loaded_agent_params['state_size'],
        action_size=loaded_agent_params['action_size'],
        alpha=loaded_agent_params['alpha'],
        gamma=loaded_agent_params['gamma'],
        epsilon=0.0,  # Set epsilon to 0 for full exploitation
        epsilon_min=loaded_agent_params['epsilon_min'],
        epsilon_decay=loaded_agent_params['epsilon_decay']
    )
    agent.q_table = defaultdict(float, loaded_q_table)
    logging.info(f"Initialized RLAgent with state size {agent.state_size} and action size {agent.action_size}.")
    logging.info(f"Set epsilon to {agent.epsilon} for evaluation (fully exploit).")

    # Run a single episode using the loaded model
    score = run_episode(agent, args.configuration_file, args.test_duration, args.random)

    # Log the final score
    if score is not None:
        logging.info(f"Episode completed. Final score: {score}")
    else:
        logging.warning("Episode failed to complete.")

# --------------------- Entry Point ---------------------

if __name__ == '__main__':
    main()
