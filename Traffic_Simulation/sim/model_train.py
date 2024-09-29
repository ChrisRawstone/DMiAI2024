from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from trafficenv import TrafficEnv
from multiprocessing import Process, Queue
from time import sleep, time
from environment import load_and_run_simulation

def train_agent_PPO():
    print("[TRAINING] Starting training process.")

    # Set up constants for the simulation
    test_duration_seconds = 30  # 5 minutes for each training episode
    configuration_file = "models/1/glue_configuration.yaml"
    random = True
    start_time = time()

    # Set up input and output queues for the simulation
    input_queue = Queue()
    output_queue = Queue()
    error_queue = Queue()

    # Start the simulation in a separate process
    p = Process(target=load_and_run_simulation, args=(configuration_file, start_time, test_duration_seconds, random, input_queue, output_queue, error_queue))
    p.start()

    # Give time for SUMO to initialize properly
    sleep(5)  # Adjust this sleep time based on your environment

    try:
        # Initialize the TrafficEnv with the queues
        env = TrafficEnv(input_queue, output_queue, configuration_file)
    except RuntimeError as e:
        print(f"[ERROR] Failed to initialize TrafficEnv: {e}")
        p.terminate()
        return
    
    # Set up logging with EvalCallback
    eval_callback = EvalCallback(env, best_model_save_path='./logs/best_model/',
                                 log_path='./logs/', eval_freq=50, deterministic=True, render=False)

    # Initialize PPO model with the environment
    model = PPO("MlpPolicy", env, verbose=1)

    print(f"[TRAINING] Initialized PPO model and TrafficEnv.")

    # Train the model for a fixed number of timesteps (since maps don't change during training)
    total_timesteps = 1000  # Example timesteps for training
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Save the trained model after completing the training
    model.save("ppo_traffic_model")

    print(f"[SAVE] PPO model saved. Ready for testing with different maps.")

    # You can terminate the simulation process after training
    p.terminate()

if __name__ == "__main__":
    train_agent_PPO()
