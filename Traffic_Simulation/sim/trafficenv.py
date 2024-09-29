import gym
from gym import spaces
import numpy as np
from time import time
from collections import defaultdict
from environment import TrafficSimulationEnvHandler  # Your environment handler


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
        self.actions = self._initialize_actions()

        # Action space is based on the number of allowed green signal combinations
        self.action_space = spaces.Discrete(len(self.actions))

        # Observation space for each signal group (vehicles on legs, stopped vehicles)
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

    def _initialize_actions(self):
        """Initializes the valid actions based on the allowed green signal combinations."""
        actions = []
        for combo in self.state.allowed_green_signal_combinations:
            actions.append(combo.groups)
        return actions

    def step(self, action):
        """Execute the given action."""
        current_tick = self.state.simulation_ticks
        ticks_since_last_change = current_tick - self.last_change_tick

        if self.last_action is None:
            self.last_action = action

        if ticks_since_last_change >= self.change_duration:
            selected_combination = self.actions[action]

            # Set signals to red initially
            next_signals = {signal.name: 'red' for signal in self.state.signals}

            # Set the selected signal combination to green
            for signal in selected_combination:
                next_signals[signal] = 'green'

            # Update current action and tick
            self.last_action = action
            self.last_change_tick = current_tick

            # Send the new signal states
            self.input_queue.put(next_signals)

        else:
            # Replay last action
            selected_combination = self.actions[self.last_action]
            next_signals = {signal.name: 'red' for signal in self.state.signals}
            for signal in selected_combination:
                next_signals[signal] = 'green'
            self.input_queue.put(next_signals)

        # Get new state
        self.state = self.output_queue.get()

        # Extract new state and calculate reward
        observation = self._extract_state(self.state)
        reward = self._calculate_reward(self.state)

        # Check if simulation is done
        done = self.state.is_terminated

        return observation, reward, done, {}

    def _extract_state(self, state_payload):
        """Extract the observation (vehicle counts, stopped vehicles) from the payload."""
        vehicle_counts = {leg: 0 for leg in self.signal_groups}
        stopped_vehicles = {leg: 0 for leg in self.signal_groups}

        for vehicle in state_payload.vehicles:
            if vehicle.leg in vehicle_counts:
                vehicle_counts[vehicle.leg] += 1
            if vehicle.speed < 1.0 and vehicle.leg in stopped_vehicles:
                stopped_vehicles[vehicle.leg] += 1

        state_vector = []
        for leg in self.signal_groups:
            state_vector.append(vehicle_counts[leg])
            state_vector.append(stopped_vehicles[leg])

        return np.array(state_vector)

    def _calculate_reward(self, state_payload):
        """Calculate reward based on vehicle delays."""
        total_stopped_vehicles = sum(1 for vehicle in state_payload.vehicles if vehicle.speed < 1.0)
        return -total_stopped_vehicles

    def reset(self):
        """No reset, return to the same state to handle termination cleanly."""
        print(f"[RESET] Reset requested, starting new simulation.")
        self.last_change_tick = 0
        self.state = self._get_state_with_timeout()

        if self.state is None:
            raise RuntimeError("[ERROR] Could not receive initial state from simulation.")

        print(f"[RESET] Received new state after reset: {self.state}")
        return self._extract_state(self.state)

    def render(self, mode='human'):
        pass
