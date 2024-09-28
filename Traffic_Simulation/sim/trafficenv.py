import gym
from gym import spaces
import numpy as np
from time import time
from collections import defaultdict
from environment import TrafficSimulationEnvHandler  # Import your environment handler with _calculate_score

# Function to compute unique valid actions from allowed combinations
def compute_unique_valid_actions(allowed_combinations):
    actions = set()  # Use a set to automatically handle duplicates

    # Extract pairs of signals from allowed combinations
    for combo in allowed_combinations:
        for group in combo.groups:
            # Sort the pair to avoid duplicates in reverse order (e.g., ['A1', 'A2'] and ['A2', 'A1'])
            pair = tuple(sorted([combo.name, group]))
            actions.add(pair)

    # Convert the set back to a list of actions
    return [list(action) for action in actions]

class TrafficEnv(gym.Env):
    def __init__(self, input_queue, output_queue, change_duration=15):
        super(TrafficEnv, self).__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.change_duration = change_duration
        self.last_action = None
        self.last_change_time = time()
        
        # Get the initial state from the output queue (from the running instance of TrafficSimulationEnvHandler)
        self.state = self._get_state_with_timeout()
        
        if self.state is None:
            raise RuntimeError("[ERROR] Could not receive initial state from simulation.")

        self.signal_groups = self.state.signal_groups
        
        # Initialize the allowed green signal combinations using the existing method
        self.actions = self.initialize_actions()
        
        # Action space is defined based on allowed green signal combinations
        self.action_space = spaces.Discrete(len(self.actions))

        # Observation space should account for total and stopped vehicles for each signal group
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(len(self.signal_groups) * 2,), dtype=np.float32
        )

    def _get_state_with_timeout(self, timeout=5):
        try:
            state = self.output_queue.get(timeout=timeout)  # Use the output queue to receive state
            print(f"[STATE] Received state: {state}")
            return state
        except Exception as e:
            print(f"[ERROR] Timeout or error while getting state: {e}")
            return None

    def initialize_actions(self):
        """
        Automatically create unique actions by combining the name and groups from the allowed green signal combinations,
        and eliminating redundancies and duplicates.
        """
        return compute_unique_valid_actions(self.state.allowed_green_signal_combinations)

    def step(self, action):
        current_tick = self.state.simulation_ticks
        ticks_since_last_change = current_tick - self.last_change_tick

        if self.last_action is None:
            self.last_action = action

        if ticks_since_last_change >= self.change_duration:
            selected_combination = self.actions[action]

            next_signals = {signal.name: 'red' for signal in self.state.signals}
            for signal in selected_combination:
                next_signals[signal] = 'green'

            # Send the new signal state to the input queue
            self.input_queue.put(next_signals)

        else:
            selected_combination = self.actions[self.last_action]
            next_signals = {signal.name: 'red' for signal in self.state.signals}
            for signal in selected_combination:
                next_signals[signal] = 'green'

            self.input_queue.put(next_signals)

        self.state = self.output_queue.get()  # Get the updated state from the output queue

        self.current_state = self.extract_state(self.state)

        reward = self.calculate_reward(self.state)

        done = self.state.simulation_ticks >= 600

        return self.current_state, reward, done, {}

    def extract_state(self, state_payload):
        leg_vehicle_count = {leg: 0 for leg in self.signal_groups}
        leg_stopped_vehicle_count = {leg: 0 for leg in self.signal_groups}
        
        for vehicle in state_payload.vehicles:
            if vehicle.leg in leg_vehicle_count:
                leg_vehicle_count[vehicle.leg] += 1
            if vehicle.speed < 1.0 and vehicle.leg in leg_stopped_vehicle_count:
                leg_stopped_vehicle_count[vehicle.leg] += 1
        
        state_vector = []
        for leg in self.signal_groups:
            state_vector.append(leg_vehicle_count[leg])
            state_vector.append(leg_stopped_vehicle_count[leg])
        
        return np.array(state_vector)

    def calculate_reward(self, state_payload):
        # The score is already calculated by TrafficSimulationEnvHandler and sent as part of the state.
        reward = state_payload.total_score

        # Prevent NaN or infinite values in the reward
        if np.isnan(reward) or np.isinf(reward):
            reward = 0
        return reward

    def reset(self):
        self.input_queue.put({"command": "reset"})
        self.state = self.output_queue.get(timeout=5)
        self.last_change_tick = self.state.simulation_ticks
        next_signals = {signal.name: 'red' for signal in self.state.signals}
        self.input_queue.put(next_signals)
        return self.extract_state(self.state)


