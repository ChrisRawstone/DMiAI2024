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

        # Initialize signal groups and states
        self.action_pairs = [
            ['A1', 'A1LeftTurn'],
            ['A2', 'A2LeftTurn'],
            ['B1', 'B1LeftTurn'],
            ['B2', 'B2LeftTurn']
        ]
        self.signal_groups = {tuple(pair): SignalGroup(pair) for pair in self.action_pairs}
        self.current_active_group = None

        # Define action space
        self.action_space = spaces.Discrete(len(self.action_pairs))

        # Define observation space
        obs_len = (3 * self.num_legs) + (2 * self.num_signals)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(obs_len,), dtype=np.float32)

        # Initialize other variables
        self.previous_total_score = 0.0

    def step(self, action):
        current_tick = self.state.simulation_ticks

        # Determine the selected signal group
        selected_group = tuple(self.action_pairs[action])

        # Check if we can switch to the new group
        if self.current_active_group != selected_group:
            can_switch = True
            if self.current_active_group is not None:
                # Check if minimum green time has been met for current group
                current_group_obj = self.signal_groups[self.current_active_group]
                can_switch = current_group_obj.can_switch(current_tick, min_green_time=10)

            if can_switch:
                # Switch to the new group
                if self.current_active_group is not None:
                    self.signal_groups[self.current_active_group].start_transition_to_red(current_tick)
                self.signal_groups[selected_group].start_transition_to_green(current_tick)
                self.current_active_group = selected_group

        # Update signal groups
        for group_obj in self.signal_groups.values():
            group_obj.update(current_tick)

        # Prepare the signal commands to send to the simulation
        next_signals = []
        for signal_name in self.signal_names:
            # Determine the state of each signal
            state = 'red'
            for group, group_obj in self.signal_groups.items():
                if signal_name in group:
                    state = group_obj.get_signal_state(signal_name)
                    break
            next_signals.append(SignalDto(name=signal_name, state=state))

        # Send the signal updates to the simulation
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
            self.input_queue.put('TERMINATE')
            self.process.join(timeout=10)
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

        # Reset signal groups
        self.signal_groups = {tuple(pair): SignalGroup(pair) for pair in self.action_pairs}
        self.current_active_group = None

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
        for signal_name in self.signal_names:
            # Determine the state and duration
            state = 0.0  # Default to red
            duration = 0
            for group_obj in self.signal_groups.values():
                if signal_name in group_obj.signals:
                    state_str = group_obj.get_signal_state(signal_name)
                    state = 1.0 if state_str == 'green' else 0.0
                    duration = group_obj.state_duration
                    break
            obs.extend([state, duration])

        return np.array(obs, dtype=np.float32)

    def render(self):
        # Rendering is not implemented
        pass

    def close(self):
        # Terminate the simulation process
        if self.process.is_alive():
            self.input_queue.put('TERMINATE')
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()

class SignalGroup:
    def __init__(self, signals):
        self.signals = signals  # List of signal names
        self.state = 'red'      # Initial state
        self.state_duration = 0
        self.last_change_time = 0

    def update(self, current_tick):
        self.state_duration = current_tick - self.last_change_time
        # Handle state transitions based on durations
        if self.state == 'redamber' and self.state_duration >= 2:
            self.state = 'green'
            self.last_change_time = current_tick
            self.state_duration = 0
        elif self.state == 'amber' and self.state_duration >= 4:
            self.state = 'red'
            self.last_change_time = current_tick
            self.state_duration = 0

    def start_transition_to_green(self, current_tick):
        self.state = 'redamber'
        self.last_change_time = current_tick
        self.state_duration = 0

    def start_transition_to_red(self, current_tick):
        self.state = 'amber'
        self.last_change_time = current_tick
        self.state_duration = 0

    def can_switch(self, current_tick, min_green_time):
        if self.state == 'green' and (current_tick - self.last_change_time) >= min_green_time:
            return True
        return False

    def get_signal_state(self, signal_name):
        return self.state
