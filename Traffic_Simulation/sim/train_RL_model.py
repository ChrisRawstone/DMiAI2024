import pickle
import multiprocessing
from multiprocessing import Process, Queue, current_process, Manager
from time import sleep, time
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List

from environment import load_and_run_simulation  # Ensure this module is accessible

import logging
import os
import sys

# Configure logging for the main process
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

@dataclass
class EnhancedState:
    vehicle_counts: List[int]
    average_speeds: List[int]
    waiting_times: List[int]
    current_signal_states: List[str]
    elapsed_time_since_last_action: int

class RLAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.q_table = defaultdict(float)  # mapping from state-action to value
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def get_state_key(self, state: EnhancedState):
        # Discretize average speeds and waiting times
        speed_bins = np.digitize(state.average_speeds, bins=[0, 5, 10, 15, 20, 25, 30])
        waiting_time_bins = np.digitize(state.waiting_times, bins=[0, 10, 20, 30, 40, 50, 60])

        return (
            tuple(state.vehicle_counts),
            tuple(speed_bins),
            tuple(waiting_time_bins),
            tuple(state.current_signal_states),
            state.elapsed_time_since_last_action
        )

    def choose_action(self, state: EnhancedState):
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

    def update_q_value(self, state: EnhancedState, action, reward, next_state: EnhancedState):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        q_predict = self.q_table[(state_key, action)]
        next_q_values = [self.q_table[(next_state_key, a)] for a in range(self.action_size)]
        q_target = reward + self.gamma * max(next_q_values)
        self.q_table[(state_key, action)] = q_predict + self.alpha * (q_target - q_predict)
        logging.debug(
            f"Updated Q-value for state {state_key} and action {action}: "
            f"{self.q_table[(state_key, action)]:.2f} (Reward: {reward}, Q_target: {q_target:.2f})"
        )

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def generate_allowed_configurations(allowed_green_signal_combinations):
    # Predefine phases with a minimum duration
    configurations = [
        {'signals': ['A1', 'A2'], 'min_duration': 30},  # Phase 0
        {'signals': ['B1', 'B2'], 'min_duration': 30},  # Phase 1
        {'signals': ['A1LeftTurn', 'A2LeftTurn'], 'min_duration': 20},  # Phase 2
        {'signals': ['B1LeftTurn', 'B2LeftTurn'], 'min_duration': 20},  # Phase 3
    ]
    return configurations

def get_enhanced_state_representation(state, signal_names, last_action_time, simulation_ticks):
    vehicle_counts = []
    average_speeds = []
    waiting_times = []

    for signal_name in signal_names:
        vehicles_in_lane = [v for v in state.vehicles if v.leg == signal_name]
        stopped_vehicles = [v for v in vehicles_in_lane if v.speed < 0.5]
        vehicle_counts.append(len(stopped_vehicles))

        if vehicles_in_lane:
            avg_speed = sum(v.speed for v in vehicles_in_lane) / len(vehicles_in_lane)
            average_speeds.append(int(avg_speed))

            # Safely compute average waiting time
            waiting_time_values = []
            for v in stopped_vehicles:
                if v.speed > 0:
                    try:
                        waiting_time = simulation_ticks - (v.distance_to_stop / v.speed)
                    except ZeroDivisionError:
                        # This should not occur due to the v.speed > 0 check, but added as a safeguard
                        waiting_time = simulation_ticks
                        logging.debug(f"ZeroDivisionError encountered for vehicle {v}. Assigned waiting_time = {waiting_time}")
                else:
                    # Assign a default high waiting time for vehicles that are completely stopped
                    waiting_time = simulation_ticks  # or another appropriate value
                    logging.debug(f"Vehicle {v} is completely stopped. Assigned waiting_time = {waiting_time}")
                waiting_time_values.append(waiting_time)

            avg_waiting_time = sum(waiting_time_values) / len(stopped_vehicles) if stopped_vehicles else 0
            waiting_times.append(int(avg_waiting_time))
        else:
            average_speeds.append(0)
            waiting_times.append(0)

    current_signal_states = [signal.state for signal in state.signals]
    elapsed_time = simulation_ticks - last_action_time

    return EnhancedState(
        vehicle_counts=vehicle_counts,
        average_speeds=average_speeds,
        waiting_times=waiting_times,
        current_signal_states=current_signal_states,
        elapsed_time_since_last_action=elapsed_time
    )

def compute_reward(previous_total_score, current_total_score, rapid_switch_penalty=5):
    """
    Compute the reward based on the change in total score.
    Positive reward if the score has decreased (better performance).
    Negative reward if the score has increased or if there's rapid switching.
    """
    # Reward is the reduction in total_score (since we want to minimize it)
    reward = previous_total_score - current_total_score

    # Penalize if current_total_score increased (i.e., more waiting)
    if reward < 0:
        reward -= rapid_switch_penalty  # Additional penalty for increased waiting

    return reward

def run_simulation(configuration_file, test_duration_seconds=600, random_flag=True):
    start_time = time()

    input_queue = Queue()
    output_queue = Queue()
    error_queue = Queue()

    p = Process(target=load_and_run_simulation, args=(
        configuration_file,
        start_time,
        test_duration_seconds,
        random_flag,
        input_queue,
        output_queue,
        error_queue
    ))

    p.start()

    # Wait for the simulation to start
    sleep(0.2)

    return p, input_queue, output_queue, error_queue

def worker_main(worker_id, num_episodes, configuration_file, test_duration_seconds, random_flag,
               model_save_dir, state_size, action_size, shared_best_score, lock):
    # Configure logging for the worker
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(f'%(asctime)s [Worker {worker_id}] [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(handler)

    # Initialize RLAgent
    agent = RLAgent(state_size, action_size)
    logger.info(f"Initialized RLAgent with state size {state_size} and action size {action_size}")

    for episode in range(1, num_episodes + 1):
        logger.info(f"Starting Episode {episode}/{num_episodes}")
        try:
            p, input_queue, output_queue, error_queue = run_simulation(configuration_file, test_duration_seconds, random_flag)

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
                    logger.warning("No state received. Terminating simulation.")
                    p.terminate()
                    p.join()
                    break

                if hasattr(state, 'is_terminated') and state.is_terminated:
                    p.join()
                    break

                simulation_ticks = state.simulation_ticks

                if not agent_initialized:
                    # Initialize the agent
                    signal_names = [signal.name for signal in state.signals]
                    allowed_configurations = generate_allowed_configurations(state.allowed_green_signal_combinations)
                    # state_size and action_size are passed from main
                    agent_initialized = True
                    logger.info("RL Agent initialized.")

                current_state = get_enhanced_state_representation(state, signal_names, last_action_time, simulation_ticks)

                # Enforce minimum duration between actions (e.g., 30 ticks)
                MIN_DURATION = 30
                if current_state.elapsed_time_since_last_action < MIN_DURATION:
                    action_index = previous_action if previous_action is not None else np.random.choice(agent.action_size)
                    logger.debug(f"Continuing with previous action {action_index} due to minimum duration constraint.")
                else:
                    action_index = agent.choose_action(current_state)
                    last_action_time = simulation_ticks  # Update the last action time

                selected_configuration = allowed_configurations[action_index]['signals']
                min_duration = allowed_configurations[action_index]['min_duration']
                logger.debug(
                    f"Tick {simulation_ticks}: Selected action {action_index} with configuration "
                    f"{selected_configuration} and min_duration {min_duration}"
                )

                prediction = {"signals": []}
                for signal_name in signal_names:
                    if signal_name in selected_configuration:
                        prediction["signals"].append({"name": signal_name, "state": "green"})
                    else:
                        prediction["signals"].append({"name": signal_name, "state": "red"})

                # Update the desired phase of the traffic lights
                next_signals = {signal['name']: signal['state'] for signal in prediction['signals']}
                input_queue.put(next_signals)

                # Compute reward if possible
                if previous_state is not None and previous_action is not None:
                    reward = compute_reward(previous_total_score, state.total_score)
                    agent.update_q_value(previous_state, previous_action, reward, current_state)
                    agent.decay_epsilon()

                previous_state = current_state
                previous_action = action_index
                previous_total_score = state.total_score

            # End of simulation, return the score
            if hasattr(state, 'total_score'):
                if state.total_score == 0:
                    state.total_score = 1e9  # Assign a high score if zero
                logger.info(f"Episode {episode} Final inverted score: {state.total_score}")

                # Check against the shared best score
                with lock:
                    if state.total_score < shared_best_score[0]:
                        shared_best_score[0] = state.total_score
                        logger.info(f"New global best score achieved: {shared_best_score[0]}. Saving model...")

                        # Define filenames with worker ID, episode number, and score
                        timestamp = int(time())
                        q_table_filename = os.path.join(model_save_dir,
                                                        f'q_table_worker_{worker_id}_episode_{episode}_score_{shared_best_score[0]}_{timestamp}.pkl')
                        agent_params_filename = os.path.join(model_save_dir,
                                                            f'agent_parameters_worker_{worker_id}_episode_{episode}_score_{shared_best_score[0]}_{timestamp}.pkl')

                        # Save the trained Q-table
                        with open(q_table_filename, 'wb') as f:
                            pickle.dump(dict(agent.q_table), f)
                        logger.info(f"Q-table saved to '{q_table_filename}'.")

                        # Save additional agent parameters if needed
                        agent_parameters = {
                            'state_size': agent.state_size,
                            'action_size': agent.action_size,
                            'alpha': agent.alpha,
                            'gamma': agent.gamma,
                            'epsilon': agent.epsilon,
                            'epsilon_min': agent.epsilon_min,
                            'epsilon_decay': agent.epsilon_decay
                        }
                        with open(agent_params_filename, 'wb') as f:
                            pickle.dump(agent_parameters, f)
                        logger.info(f"Agent parameters saved to '{agent_params_filename}'.")
            else:
                logger.warning(f"Episode {episode} failed to retrieve total_score.")

            logger.info(f"Completed Episode {episode}/{num_episodes}")

        except Exception as e:
            logger.error(f"An error occurred during Episode {episode}: {e}")

    logger.info(f"Worker {worker_id} completed all {num_episodes} episodes.")

def main():
    num_episodes = 300  # Total number of training episodes
    configuration_file = "models/1/glue_configuration.yaml"
    test_duration_seconds = 300  # Duration per simulation
    random_flag = True  # Set to True or False as needed

    # Determine the number of CPU cores
    num_cores = multiprocessing.cpu_count()
    logging.info(f"Number of CPU cores available: {num_cores}")

    # Determine the number of episodes per worker
    episodes_per_worker = num_episodes // num_cores
    remaining_episodes = num_episodes % num_cores

    # Create a directory to save improved models
    model_save_dir = "improved_models"
    os.makedirs(model_save_dir, exist_ok=True)

    # Initialize Manager for shared best score and lock
    manager = Manager()
    shared_best_score = manager.list([float('inf')])  # Initialize with infinity
    lock = manager.Lock()

    # Perform a temporary run to determine state_size and action_size
    logging.info("Performing a temporary simulation run to determine state and action sizes...")
    temp_p, temp_input_queue, temp_output_queue, temp_error_queue = run_simulation(configuration_file, test_duration_seconds, random_flag)
    sleep(0.2)
    try:
        temp_state = temp_output_queue.get(timeout=10)
    except Exception as e:
        logging.error(f"Failed to retrieve initial state: {e}. Exiting.")
        temp_p.terminate()
        temp_p.join()
        return

    if not hasattr(temp_state, 'signals') or not hasattr(temp_state, 'allowed_green_signal_combinations') or not hasattr(temp_state, 'total_score'):
        logging.error("Initial state is missing required attributes. Exiting.")
        temp_p.terminate()
        temp_p.join()
        return

    signal_names = [signal.name for signal in temp_state.signals]
    allowed_configurations = generate_allowed_configurations(temp_state.allowed_green_signal_combinations)
    state_size = len(signal_names) * 3 + len(temp_state.signals) + 1  # Example size calculation
    action_size = len(allowed_configurations)

    logging.info(f"Determined state_size: {state_size}, action_size: {action_size}")

    # Initialize RLAgent for temporary run (not used further)
    temp_agent = RLAgent(state_size, action_size)

    temp_p.terminate()
    temp_p.join()

    # Calculate episodes per worker, distributing the remaining episodes
    workers = []
    for i in range(num_cores):
        # Distribute remaining episodes among the first few workers
        assigned_episodes = episodes_per_worker + (1 if i < remaining_episodes else 0)
        if assigned_episodes == 0:
            continue  # Skip if no episodes are assigned

        worker = Process(
            target=worker_main,
            args=(
                i + 1,  # Worker ID starting from 1
                assigned_episodes,
                configuration_file,
                test_duration_seconds,
                random_flag,
                model_save_dir,
                state_size,
                action_size,
                shared_best_score,
                lock
            )
        )
        workers.append(worker)
        worker.start()
        logging.info(f"Started Worker {i + 1} with {assigned_episodes} episodes.")

    # Wait for all workers to finish
    for worker in workers:
        worker.join()
        logging.info(f"Worker {worker.pid} has finished.")

    logging.info("All workers have completed their episodes.")

    # Optionally, aggregate results here
    # For example, find the best Q-table among all saved models
    # This part can be customized based on specific needs

    # Log the global best score achieved
    logging.info(f"Global Best Score Achieved: {shared_best_score[0]}")

if __name__ == '__main__':
    main()
