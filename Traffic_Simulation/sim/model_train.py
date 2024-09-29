from stable_baselines3 import PPO
from trafficenv import TrafficEnv
from multiprocessing import Process, Queue
from time import sleep, time
from environment import load_and_run_simulation


def start_new_simulation(configuration_file, test_duration_seconds, random, input_queue, output_queue, error_queue):
    """
    Helper function to start a new SUMO simulation in a separate process.
    """
    start_time = time()
    p = Process(target=load_and_run_simulation, args=(configuration_file, start_time, test_duration_seconds, random, input_queue, output_queue, error_queue))
    p.start()

    # Give time for SUMO to initialize properly
    sleep(2)
    
    return p


def train_agent_PPO():
    print("[TRAINING] Starting training process.")

    test_duration_seconds = 600  # Run the simulation for this long before starting a new episode
    configuration_file = "models/1/glue_configuration.yaml"
    random = True

    # Set up input and output queues for the simulation
    input_queue = Queue()
    output_queue = Queue()
    error_queue = Queue()

    # Start the first simulation in a separate process
    p = start_new_simulation(configuration_file, test_duration_seconds, random, input_queue, output_queue, error_queue)

    try:
        # Initialize the TrafficEnv with the queues
        env = TrafficEnv(input_queue, output_queue)
    except RuntimeError as e:
        print(f"[ERROR] Failed to initialize TrafficEnv: {e}")
        p.terminate()
        return

    # Initialize PPO model with the environment
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)

    print(f"[TRAINING] Initialized PPO model and TrafficEnv.")

    # Train the model for a given number of timesteps
    total_timesteps = 10000  # You can increase this for more training

    # Run the training loop without resetting the environment mid-episode
    while True:
        try:
            model.learn(total_timesteps=total_timesteps)
            break  # Exit the loop when training is done
        except RuntimeError as e:
            if 'Could not receive initial state from simulation' in str(e):
                print("[ERROR] Simulation terminated, restarting a new one.")
                p.terminate()  # Terminate the current process
                p = start_new_simulation(configuration_file, test_duration_seconds, random, input_queue, output_queue, error_queue)
                env.reset()  # Reset the environment to start a new simulation
            else:
                print(f"[ERROR] Unexpected error during training: {e}")
                p.terminate()
                break

    # Save the trained model
    model.save("ppo_traffic_model")
    print(f"[SAVE] PPO model saved.")

    # Terminate the simulation process
    p.terminate()


if __name__ == "__main__":
    train_agent_PPO()
