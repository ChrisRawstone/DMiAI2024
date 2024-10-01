import pickle
from multiprocessing import Process, Queue
from time import sleep, time
import numpy as np
from collections import defaultdict

from environment import load_and_run_simulation

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO for general information
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class RLAgent:
    def __init__(self, state_size, action_size, q_table=None, alpha=0.1, gamma=0.9, epsilon=0.0):
        self.q_table = defaultdict(float, q_table if q_table else {})  # mapping from state-action to value
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate (set to 0 for exploitation)

    def get_state_key(self, state):
        # Convert the state to a tuple to use as a key
        return tuple(state)

    def choose_action(self, state):
        # Always exploit; no exploration
        state_key = self.get_state_key(state)
        q_values = [self.q_table[(state_key, a)] for a in range(self.action_size)]
        max_q = max(q_values)
        # In case multiple actions have the same max Q-value, choose randomly among them
        actions_with_max_q = [a for a in range(self.action_size) if q_values[a] == max_q]
        action = np.random.choice(actions_with_max_q)
        logging.info(f"Chosen action {action} with Q-value {max_q} for state {state_key}")
        return action

def generate_allowed_configurations(allowed_green_signal_combinations):
    # This function generates all allowed configurations of signals
    # Each configuration is a list of signals to set to green
    configurations = []
    # Build a mapping from signal to allowed groups
    signal_to_allowed_groups = {}
    for combo in allowed_green_signal_combinations:
        signal = combo.name
        allowed_groups = combo.groups
        signal_to_allowed_groups[signal] = allowed_groups

    # For simplicity, define some phases manually based on allowed combinations
    configurations = [
        ['A1', 'A2'],  # Phase 0
        ['B1', 'B2'],  # Phase 1
        ['A1LeftTurn', 'A2LeftTurn'],  # Phase 2
        ['B1LeftTurn', 'B2LeftTurn'],  # Phase 3
    ]
    return configurations

def get_state_representation(state, signal_names):
    # For each signal, count the number of vehicles waiting (speed less than a threshold, e.g., 0.1)
    vehicle_counts = []
    for signal_name in signal_names:
        count = 0
        for vehicle in state.vehicles:
            if vehicle.leg == signal_name:
                if vehicle.speed < 0.1:
                    count += 1
        # Discretize the count into bins: 0, 1-2, 3+
        if count == 0:
            bin_value = 0
        elif count <= 2:
            bin_value = 1
        else:
            bin_value = 2
        vehicle_counts.append(bin_value)
    return vehicle_counts

def compute_reward(state):
    count = 0
    for vehicle in state.vehicles:
        if vehicle.speed < 0.1:
            count += 1
    # We can define the reward as negative of the count
    reward = -count
    return reward

def run_simulation(configuration_file, test_duration_seconds=600, random=True):
    start_time = time()

    input_queue = Queue()
    output_queue = Queue()
    error_queue = Queue()
    errors = []

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

    # For logging
    actions = {}

    return p, input_queue, output_queue, error_queue, actions

def run_trained_simulation(agent, configuration_file="models/1/glue_configuration.yaml", test_duration_seconds=600, random=True):
    p, input_queue, output_queue, error_queue, actions = run_simulation(configuration_file, test_duration_seconds, random)

    previous_state = None
    previous_action = None
    signal_names = []
    allowed_configurations = []

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

        if not signal_names:
            # Initialize configurations based on the first state
            signal_names = [signal.name for signal in state.signals]
            allowed_configurations = generate_allowed_configurations(state.allowed_green_signal_combinations)
            logging.info("Loaded configurations based on simulation state.")

        current_state = get_state_representation(state, signal_names)

        action_index = agent.choose_action(current_state)
        selected_configuration = allowed_configurations[action_index]
        logging.info(f"Tick {state.simulation_ticks}: Selected action {action_index} with configuration {selected_configuration}")

        prediction = {}
        prediction["signals"] = []
        for signal_name in signal_names:
            if signal_name in selected_configuration:
                prediction["signals"].append({"name": signal_name, "state": "green"})
            else:
                prediction["signals"].append({"name": signal_name, "state": "red"})

        # Update the desired phase of the traffic lights
        next_signals = {}
        current_tick = state.simulation_ticks

        for signal in prediction['signals']:
            actions[current_tick] = (signal['name'], signal['state'])
            next_signals[signal['name']] = signal['state']

        signal_logic_errors = input_queue.put(next_signals)

        if signal_logic_errors:
            logging.error(f"Signal logic errors: {signal_logic_errors}")

    # End of simulation, return the score

    # Transform the score to the range [0, 1]
    if state.total_score == 0:
        state.total_score = 1e9

    logging.info(f"Final inverted score: {state.total_score}")

    return state.total_score

def main():
    # Load the trained Q-table
    with open('q_table.pkl', 'rb') as f:
        q_table = pickle.load(f)
    logging.info("Q-table loaded from 'q_table.pkl'.")

    # Load agent parameters
    try:
        with open('agent_parameters.pkl', 'rb') as f:
            agent_parameters = pickle.load(f)
        logging.info("Agent parameters loaded from 'agent_parameters.pkl'.")
    except FileNotFoundError:
        logging.error("Agent parameters file not found. Ensure 'agent_parameters.pkl' exists.")
        return

    # Initialize RL agent with loaded Q-table and parameters
    agent = RLAgent(
        state_size=agent_parameters['state_size'],
        action_size=agent_parameters['action_size'],
        q_table=q_table,
        alpha=agent_parameters['alpha'],
        gamma=agent_parameters['gamma'],
        epsilon=0.0  # No exploration; pure exploitation
    )
    logging.info("RLAgent initialized with loaded Q-table.")

    configuration_file = "models/1/glue_configuration.yaml"
    test_duration_seconds = 600  # Duration for the simulation

    # Run the simulation using the trained agent
    score = run_trained_simulation(agent, configuration_file, test_duration_seconds, random=False)
    logging.info(f"Trained simulation completed with score: {score}")

if __name__ == '__main__':
    main()
