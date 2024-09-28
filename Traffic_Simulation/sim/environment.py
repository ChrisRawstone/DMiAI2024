import gym
from gym import spaces
import numpy as np
from time import time
from collections import defaultdict

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
        
        # Get the initial state
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
            state = self.output_queue.get(timeout=timeout)
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
        """
        Apply the action based on the combination of the name and groups.
        Lock the action for the duration (using simulation ticks), and replay the last action if it's still locked.
        """
        current_tick = self.state.simulation_ticks
        ticks_since_last_change = current_tick - self.last_change_tick

        # If this is the first step, set last_action to the current action
        if self.last_action is None:
            self.last_action = action

        # Only allow signal change if enough simulation ticks have passed
        if ticks_since_last_change >= self.change_duration:
            # Apply the selected allowed green light combination
            selected_combination = self.actions[action]

            # Set all signals to red first
            next_signals = {signal.name: 'red' for signal in self.state.signals}

            # Set the selected combination signals to green
            for signal in selected_combination:
                next_signals[signal] = 'green'

            # Update the current signal combination and reset the tick timer
            self.last_action = action  # Store the last action taken
            self.last_change_tick = current_tick  # Reset the tick timer

            # Send new signal states to the input queue
            self.input_queue.put(next_signals)

        else:
            # Replay the last action (keep the last signal configuration)
            selected_combination = self.actions[self.last_action]

            # Set all signals to red first
            next_signals = {signal.name: 'red' for signal in self.state.signals}

            # Keep the last selected combination signals green
            for signal in selected_combination:
                next_signals[signal] = 'green'

            # Resend the same signal states to the input queue
            self.input_queue.put(next_signals)

        # Advance the simulation step and get the new state
        self.state = self.output_queue.get()

        # Extract the new state (e.g., vehicle count, signal status)
        self.current_state = self.extract_state(self.state)

        # Calculate reward based on traffic performance (e.g., minimize vehicle waiting time)
        reward = self.calculate_reward(self.state)

        # Simulation done after a fixed number of ticks (10 minutes or 600 ticks)
        done = self.state.simulation_ticks >= 600

        return self.current_state, reward, done, {}

    def extract_state(self, state_payload):
        """
        Extract state from the payload based on vehicle counts and their speed.
        We'll count the number of vehicles per leg and how many are 'stopped'.
        """
        # Create a dictionary to count vehicles per leg
        leg_vehicle_count = {leg: 0 for leg in self.signal_groups}
        leg_stopped_vehicle_count = {leg: 0 for leg in self.signal_groups}
        
        # Loop through all vehicles in the state
        for vehicle in state_payload.vehicles:
            # Count total vehicles for each leg
            if vehicle.leg in leg_vehicle_count:
                leg_vehicle_count[vehicle.leg] += 1
            
            # Count stopped or very slow vehicles (speed < 1.0)
            if vehicle.speed < 1.0 and vehicle.leg in leg_stopped_vehicle_count:
                leg_stopped_vehicle_count[vehicle.leg] += 1
        
        # Convert counts into a state vector, e.g., [total_vehicles_A1, stopped_vehicles_A1, total_vehicles_B1, ...]
        state_vector = []
        for leg in self.signal_groups:
            state_vector.append(leg_vehicle_count[leg])
            state_vector.append(leg_stopped_vehicle_count[leg])
        
        return np.array(state_vector)

    def calculate_reward(self, state_payload):
        """
        Calculate the reward based on traffic efficiency.
        Penalize the agent based on the number of vehicles that are stopped or moving slowly.
        """
        total_stopped_vehicles = 0
        
        # Count all vehicles with speed < 1.0, considering them as "stopped"
        for vehicle in state_payload.vehicles:
            if vehicle.speed < 1.0:
                total_stopped_vehicles += 1
        
        # Negative reward proportional to the number of stopped vehicles
        reward = -total_stopped_vehicles
        
        return reward

    def reset(self):
        print(f"[RESET] Attempting to reset the environment.")

        try:
            # Optionally, send a reset command to the simulation
            self.input_queue.put({"command": "reset"})  # If your simulation accepts reset commands

            # Wait for the simulation to reset
            self.state = self.output_queue.get(timeout=5)
            print(f"[RESET] Received new state after reset: {self.state}")

            if self.state.is_terminated:
                print("[RESET] Simulation has terminated and cannot be reset.")
                return None

            # Reset any internal state here
            self.last_change_tick = self.state.simulation_ticks  # Reset the tick timer

            # Ensure vehicles and signals are reset
            if len(self.state.vehicles) > 0:
                print(f"[RESET WARNING] Vehicles were not reset properly. Clearing manually.")
                self.state.vehicles.clear()  # Ensure the vehicles list is cleared

            # Reset signals to red
            next_signals = {signal.name: 'red' for signal in self.state.signals}
            self.input_queue.put(next_signals)

            print(f"[RESET] Successfully reset the environment.")

            # Extract the initial state
            return self.extract_state(self.state)

        except Exception as e:
            print(f"[ERROR] Timeout or error while resetting environment: {e}")
            return None


    def render(self, mode='human'):
        pass
