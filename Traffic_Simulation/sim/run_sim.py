from multiprocessing import Process, Queue
from time import sleep, time
import numpy as np
from collections import defaultdict

from environment import load_and_run_simulation

import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class RLAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.q_table = defaultdict(float)  # mapping from state-action to value
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def get_state_key(self, state):
        # Convert the state to a tuple to use as a key
        return tuple(state)

    def choose_action(self, state):
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

    def update_q_value(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        q_predict = self.q_table[(state_key, action)]
        next_q_values = [self.q_table[(next_state_key, a)] for a in range(self.action_size)]
        q_target = reward + self.gamma * max(next_q_values)
        self.q_table[(state_key, action)] = q_predict + self.alpha * (q_target - q_predict)
        logging.debug(f"Updated Q-value for state {state_key} and action {action}: {self.q_table[(state_key, action)]:.2f} (Reward: {reward}, Q_target: {q_target:.2f})")


    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



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

def run_game():

    test_duration_seconds = 600
    random = True
    configuration_file = "models/1/glue_configuration.yaml"
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

    # Initialize RL agent variables
    agent_initialized = False
    previous_state = None
    previous_action = None

    while True:

        state = output_queue.get()

        if state.is_terminated:
            p.join()
            break

        if not agent_initialized:
            # Initialize the agent
            signal_names = [signal.name for signal in state.signals]
            allowed_configurations = generate_allowed_configurations(state.allowed_green_signal_combinations)
            action_size = len(allowed_configurations)
            state_size = len(signal_names)
            agent = RLAgent(state_size, action_size)
            agent_initialized = True
            logging.info("RL Agent initialized.")


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
            errors.append(signal_logic_errors)

        if previous_state is not None and previous_action is not None:
            reward = compute_reward(state)
            agent.update_q_value(previous_state, previous_action, reward, current_state)
            agent.decay_epsilon()

        previous_state = current_state
        previous_action = action_index

    # End of simulation, return the score

    # Transform the score to the range [0, 1]
    if state.total_score == 0:
        state.total_score = 1e9


    logging.info(f"Final inverted score: {state.total_score}")


    return state.total_score

if __name__ == '__main__':
    run_game()
