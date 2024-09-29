import gym
from gym import spaces
import numpy as np
from time import time
from environment import TrafficSimulationEnvHandler, load_configuration  # Import TrafficSimulationEnvHandler and load_configuration

class TrafficEnv(gym.Env):
    def __init__(self, input_queue, output_queue, config_file, change_duration=15, max_ticks=600):
        super(TrafficEnv, self).__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.change_duration = change_duration
        self.last_action = None
        self.last_change_time = time()
        self.simulation_ticks = 0  # Keep track of ticks
        self.max_ticks = max_ticks  # Max ticks for the simulation

        # **Load the simulation configuration from the file**
        self.configuration_file = config_file
        self.simulation = self._load_simulation_configuration()  # Directly assign to self.simulation

        # Debug: Check if the simulation has been properly initialized
        print(f"[DEBUG] Simulation initialized with signal groups: {self.simulation.signal_groups}")

        # Get the initial state from the simulation
        self.state = self.simulation.get_observable_state()

        self.last_action_time = None
        self.signal_minimum_times = {
            'green': 6,        # Green light minimum duration (in seconds)
            'yellow': 4,       # Yellow light minimum duration (in seconds)
            'red_yellow': 2    # Red/yellow minimum duration (in seconds)
        }
        self.current_signal_state = 'red_yellow'  # Start with red/yellow

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

    def _load_simulation_configuration(self):
        """Load the configuration details from a YAML file."""
        # Load the configuration details for the simulation
        start_time = time()
        test_duration_seconds = self.max_ticks

        # Load the configuration file and return the initialized TrafficSimulationEnvHandler
        simulation_env = load_configuration(self.configuration_file, start_time, test_duration_seconds)

        if simulation_env is None:
            raise RuntimeError("[ERROR] Failed to load simulation configuration.")
        
        # Set queues for input/output communication with the simulation
        simulation_env.set_queues(self.input_queue, self.output_queue, None)  # Assuming error queue is not used here

        return simulation_env

    def _initialize_actions(self):
        """Initializes the valid actions based on the allowed green signal combinations."""
        actions = []
        
        # Debug: Ensure that allowed_green_signal_combinations is correctly loaded
        print(f"[DEBUG] Allowed Green Signal Combinations: {self.simulation.allowed_green_signal_combinations}")

        # Iterate over the dictionary of allowed green signal combinations
        for signal, allowed_groups in self.simulation.allowed_green_signal_combinations.items():
            # Append the allowed groups as valid actions
            actions.append(allowed_groups)
        
        return actions

    def step(self, action):
        """Execute the given action and advance the simulation."""
        current_time = time()

        # Enforce minimum signal state durations
        if self.last_action_time is None:
            self.last_action_time = current_time

        elapsed_time = current_time - self.last_action_time

        # Only allow the signal to change if the minimum duration has passed for the current state
        if self.current_signal_state == 'green' and elapsed_time < self.signal_minimum_times['green']:
            action = self.last_action  # Keep the same action if the minimum green time hasn't passed
        elif self.current_signal_state == 'yellow' and elapsed_time < self.signal_minimum_times['yellow']:
            action = self.last_action  # Keep the same action if the minimum yellow time hasn't passed
        elif self.current_signal_state == 'red_yellow' and elapsed_time < self.signal_minimum_times['red_yellow']:
            action = self.last_action  # Keep the same action if the minimum red/yellow time hasn't passed
        else:
            # Update the signal state and action only when the minimum time has passed
            self.last_action = action
            self.last_action_time = current_time
            self.current_signal_state = 'green' if self.current_signal_state == 'red_yellow' else 'yellow'

        # Set signals to red initially
        next_signals = {signal.name: 'red' for signal in self.state.signals}

        # Update signal state logic here based on the action and current signal state
        selected_combination = self.actions[action]
        for signal in selected_combination:
            next_signals[signal] = 'green'

        # Send the new signal states to the simulation
        self.input_queue.put(next_signals)

        # Advance the simulation by one tick
        self.simulation._run_one_tick()

        # Get the next state after applying the action
        self.state = self.simulation.get_observable_state()

        # Extract observation and calculate reward
        observation = self._extract_state(self.state)
        reward = self._calculate_reward(self.state)

        # Update the simulation tick count
        self.simulation_ticks += 1  

        # Check if the environment has reached the maximum duration (max_ticks)
        done = self.simulation_ticks >= self.max_ticks or self.state.is_terminated

        return observation, reward, done, {}

    def reset(self):
        """Reset the environment after map change or max ticks."""
        print(f"[INFO] Resetting simulation...")

        # Reset tick counter and simulation state
        self.simulation_ticks = 0
        self.last_change_tick = 0

        # Reset the SUMO simulation via TrafficSimulationEnvHandler
        self.simulation.reset_simulation()

        # Get a fresh state from the simulation
        self.state = self.simulation.get_observable_state()

        if self.state is None:
            raise RuntimeError("[ERROR] Could not receive initial state from simulation.")

        # Update action and observation spaces based on new state
        self.actions = self._initialize_actions()
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(len(self.signal_groups) * 2,), dtype=np.float32
        )

        return self._extract_state(self.state)

    def _extract_state(self, state_payload):
        """Extracts the observation (vehicle counts, stopped vehicles) from the payload."""
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
        """Calculates the reward based on vehicle delays."""
        total_stopped_vehicles = sum(1 for vehicle in state_payload.vehicles if vehicle.speed < 1.0)
        return -total_stopped_vehicles
