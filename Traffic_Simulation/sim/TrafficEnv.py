import gymnasium as gym
import numpy as np
from multiprocessing import Process, Queue
from time import sleep, time
from environment import load_and_run_simulation
from dtos import SignalDto
from gymnasium import spaces

class TrafficEnv(gym.Env):
    def __init__(self):
        super(TrafficEnv, self).__init__()

        # Simulation parameters
        self.configuration_file = "models/1/glue_configuration.yaml"
        self.start_time = time()
        self.test_duration_seconds = 300
        self.random = True

        # Set up the queues
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.error_queue = Queue()

        # Start the simulation process
        self.process = Process(target=load_and_run_simulation, args=(
            self.configuration_file,
            self.start_time,
            self.test_duration_seconds,
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

        # Initialize signal durations
        self.signal_durations = {signal.name: 0 for signal in self.state.signals}

        # Define action space
        self.action_pairs = [
            ['A1', 'A1LeftTurn'],
            ['A2', 'A2LeftTurn'],
            ['B1', 'B1LeftTurn'],
            ['B2', 'B2LeftTurn']
        ]
        self.action_space = spaces.Discrete(len(self.action_pairs))

        # Define observation space
        # Observation: [num_cars_per_leg, num_waiting_cars_per_leg, avg_speed_per_leg] + [signal_state, signal_duration] per signal
        obs_len = (3 * self.num_legs) + (2 * self.num_signals)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(obs_len,), dtype=np.float32)

        # Initialize other variables
        self.previous_total_score = 0.0

    def step(self, action):
        # Send the action to the simulation
        active_group = self.action_pairs[action]
        next_signals = []
        for signal_name in self.signal_names:
            if signal_name in active_group:
                next_signals.append(SignalDto(name=signal_name, state="green"))
            else:
                next_signals.append(SignalDto(name=signal_name, state="red"))

        # Update signal durations and send signals to simulation
        for signal in next_signals:
            if signal.state == "green":
                self.signal_durations[signal.name] += 1
            else:
                self.signal_durations[signal.name] = 0  # Reset duration for red signals
        self.input_queue.put({signal.name: signal.state for signal in next_signals})

        # Receive the next state
        self.state = self.output_queue.get()

        # Extract observations
        obs = self._get_observation()

        # Compute reward
        reward = -(self.state.total_score - self.previous_total_score)
        self.previous_total_score = self.state.total_score

        # Check if terminated
        terminated = self.state.is_terminated

        # Since there's no truncation logic, set truncated to False
        truncated = False

        # Additional info (optional)
        info = {}

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Terminate the existing simulation process
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()

        # Reset queues
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.error_queue = Queue()

        # Start new simulation process
        self.start_time = time()
        self.process = Process(target=load_and_run_simulation, args=(
            self.configuration_file,
            self.start_time,
            self.test_duration_seconds,
            self.random,
            self.input_queue,
            self.output_queue,
            self.error_queue))
        self.process.start()

        # Wait for the simulation to start and get the initial state
        sleep(0.2)
        self.state = self.output_queue.get()

        # Reset signal durations
        self.signal_durations = {signal.name: 0 for signal in self.state.signals}

        # Reset previous total score
        self.previous_total_score = 0.0

        # Extract initial observation
        obs = self._get_observation()

        # Return the initial observation and info dictionary
        return obs, {}

    def _get_observation(self):
        # Initialize observation list
        obs = []

        # Initialize per-leg data
        leg_data = {leg_name: {'num_cars': 0, 'num_waiting': 0, 'total_speed': 0, 'num_speed_samples': 0} for leg_name in self.leg_names}

        # Process vehicle data
        for vehicle in self.state.vehicles:
            leg = vehicle.leg
            if leg not in leg_data:
                continue
            leg_data[leg]['num_cars'] += 1
            leg_data[leg]['total_speed'] += vehicle.speed
            leg_data[leg]['num_speed_samples'] += 1
            if vehicle.speed < 0.5:  # Consider vehicle waiting if speed < 0.5 m/s
                leg_data[leg]['num_waiting'] += 1

        # Append leg data to observation
        for leg_name in self.leg_names:
            data = leg_data[leg_name]
            num_cars = data['num_cars']
            num_waiting = data['num_waiting']
            avg_speed = data['total_speed'] / data['num_speed_samples'] if data['num_speed_samples'] > 0 else 0.0
            obs.extend([num_cars, num_waiting, avg_speed])

        # Append signal states and durations to observation
        for signal in self.state.signals:
            state = 1.0 if signal.state == 'green' else 0.0
            duration = self.signal_durations[signal.name]
            obs.extend([state, duration])

        return np.array(obs, dtype=np.float32)

    def render(self):
        # Rendering is not implemented
        pass

    def close(self):
        # Terminate the simulation process
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
