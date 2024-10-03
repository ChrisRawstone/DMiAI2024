import gymnasium as gym
import numpy as np
from multiprocessing import Process, Queue
from time import sleep
from environment import load_and_run_simulation
from gymnasium import spaces
from time import time

class TrafficEnv(gym.Env):
    def __init__(self):
        super(TrafficEnv, self).__init__()
        # Simulation parameters
        self.configuration_file = "models/1/glue_configuration.yaml"
        self.test_duration_ticks = 300  # Use ticks instead of seconds
        self.random = True
        
        # Set up the queues
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.error_queue = Queue()

        # Start the simulation process
        self.process = Process(target=load_and_run_simulation, args=(
            self.configuration_file,
            time(),  # Pass the current time, isn't used but required
            self.test_duration_ticks,
            self.random,
            self.input_queue,
            self.output_queue,
            self.error_queue))
        self.process.start()
        
        # Wait for the simulation to start and get the initial state
        sleep(0.2)
        self.state = self.output_queue.get()

        # Extract leg names and initialize per-leg data
        self.leg_names = [leg.name for leg in self.state.legs]
        self.num_legs = len(self.leg_names)

        # Extract signal names
        self.signal_names = [signal.name for signal in self.state.signals]
        self.num_signals = len(self.signal_names)

        # Define the allowed signal pairs
        self.signal_pairs = [
            ['A1', 'A1LeftTurn'],
            ['A2', 'A2LeftTurn'],
            ['B1', 'B1LeftTurn'],
            ['B2', 'B2LeftTurn'],
            ['A2LeftTurn', 'A1LeftTurn'],
            ['B1LeftTurn', 'B2LeftTurn']
        ]
        self.num_signal_pairs = len(self.signal_pairs)
        self.max_duration = 30  # Maximum green duration in ticks

        # Action space: choose a signal pair and a duration
        self.action_space = spaces.MultiDiscrete([self.num_signal_pairs, self.max_duration])

        # Observation space (normalized between 0 and 1)
        obs_len = (4 * self.num_legs) + (2 * self.num_signals)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32)

        # Initialize other variables
        self.previous_total_score = 0.0
        self.current_active_groups = {}
        self.current_tick = 0

    def step(self, action):
        self.current_tick = self.state.simulation_ticks

        # Unpack action
        signal_pair_idx, duration = action
        
        # Get the corresponding signal pair
        signal_pair = self.signal_pairs[signal_pair_idx]

        # Update active signal groups
        self.current_active_groups = {signal: duration for signal in signal_pair}

        # Prepare the signal commands to send to the simulation
        next_signals = {signal_name: 'green' if signal_name in self.current_active_groups else 'red' for signal_name in self.signal_names}
        self.input_queue.put(next_signals)

        # Decrease durations and remove expired signals
        expired_signals = [signal_name for signal_name, remaining in self.current_active_groups.items() if remaining <= 0]
        for signal_name in expired_signals:
            del self.current_active_groups[signal_name]

        # Receive the next state
        self.state = self.output_queue.get()

        # Extract observations
        obs = self._get_observation()

        # Compute reward
        reward = self._compute_reward()

        # Check if terminated
        terminated = self.state.is_terminated
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.process.is_alive():
            self.input_queue.put('TERMINATE')
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()

        # Restart the simulation process
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.error_queue = Queue()
        self.process = Process(target=load_and_run_simulation, args=(
            self.configuration_file,
            time(),
            self.test_duration_ticks,
            self.random,
            self.input_queue,
            self.output_queue,
            self.error_queue))
        self.process.start()
        sleep(0.2)

        self.state = self.output_queue.get()
        self.previous_total_score = 0.0
        self.current_active_groups = {}
        self.current_tick = 0
        obs = self._get_observation()

        return obs, {}

    def _get_observation(self):
        obs = []
        leg_data = {leg_name: {'num_cars': 0, 'num_waiting': 0, 'total_speed': 0.0, 'avg_distance': 0.0} for leg_name in self.leg_names}

        for vehicle in self.state.vehicles:
            leg = vehicle.leg
            if leg not in leg_data:
                continue
            leg_data[leg]['num_cars'] += 1
            leg_data[leg]['total_speed'] += vehicle.speed
            leg_data[leg]['avg_distance'] += vehicle.distance_to_stop
            if vehicle.speed < 0.5:
                leg_data[leg]['num_waiting'] += 1

        for leg_name in self.leg_names:
            data = leg_data[leg_name]
            num_cars = data['num_cars']
            num_waiting = data['num_waiting']
            avg_speed = data['total_speed'] / num_cars if num_cars > 0 else 0.0
            avg_distance = data['avg_distance'] / num_cars if num_cars > 0 else 0.0
            num_cars_normalized = min(num_cars / 50, 1.0)
            num_waiting_normalized = min(num_waiting / 50, 1.0)
            avg_speed_normalized = min(avg_speed / 25, 1.0)
            avg_distance_normalized = min(avg_distance / 500, 1.0)
            obs.extend([num_cars_normalized, num_waiting_normalized, avg_speed_normalized, avg_distance_normalized])

        for signal_name in self.signal_names:
            state = 1.0 if signal_name in self.current_active_groups else 0.0
            duration = self.current_active_groups.get(signal_name, 0)
            duration_normalized = min(duration / self.test_duration_ticks, 1.0)
            obs.extend([state, duration_normalized])

        return np.array(obs, dtype=np.float32)

    def _compute_reward(self):
        # Use the change in total_score to compute the reward
        reward = -(self.state.total_score - self.previous_total_score)
        self.previous_total_score = self.state.total_score
        return reward

    def close(self):
        if self.process.is_alive():
            self.input_queue.put('TERMINATE')
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()
