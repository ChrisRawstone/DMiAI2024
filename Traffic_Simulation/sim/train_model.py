# train_model.py

from stable_baselines3 import PPO
from trafficenv import TrafficEnv
from multiprocessing import Process, Queue
from time import time, sleep
from environment import load_and_run_simulation, TrafficSimulationEnvHandler  # Import the handler

def train_agent_PPO():
    print("[TRAINING] Starting training process.")

    # Configuration for the simulation
    test_duration_seconds = 600
    random = True
    configuration_file = "models/1/glue_configuration.yaml"
    start_time = time()

    # Initialize the input/output queues for communication
    input_queue = Queue()
    output_queue = Queue()
    error_queue = Queue()

    # Start the traffic simulation in a separate process
    p = Process(target=load_and_run_simulation, args=(configuration_file,
                                                      start_time,
                                                      test_duration_seconds,
                                                      random,
                                                      input_queue,
                                                      output_queue,
                                                      error_queue))
    
    p.start()

    # Wait for the simulation to start
    sleep(0.2)

    # Use the existing TrafficSimulationEnvHandler from the separate process (from load_and_run_simulation)
    # Instead of creating a new one, use the one already running
    try:
        # Use the input and output queues and connect to the running instance
        env = TrafficEnv(input_queue, output_queue)
    except RuntimeError as e:
        print(f"[ERROR] Failed to initialize TrafficEnv: {e}")
        p.terminate()
        return
    
    # Initialize the PPO model with the environment
    model = PPO("MlpPolicy", env, verbose=1)

    # Print initialization info
    print(f"[TRAINING] Initialized PPO model and TrafficEnv.")

    # Train the PPO model
    total_timesteps = 10000  # Adjust based on your needs
    try:
        print(f"[TRAINING] Starting PPO training for {total_timesteps} timesteps.")
        model.learn(total_timesteps=total_timesteps)
    except Exception as e:
        print(f"[ERROR] Training interrupted: {e}")
        p.terminate()
        return

    # Save the trained model
    model.save("ppo_traffic_signals")
    print(f"[SAVE] PPO model saved to 'ppo_traffic_signals'.")

    # Wait for the simulation process to finish
    p.join()
    print("[TRAINING] Training process completed.")

# Run the training
if __name__ == "__main__":
    train_agent_PPO()
