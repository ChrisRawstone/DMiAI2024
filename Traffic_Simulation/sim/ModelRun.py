from stable_baselines3 import PPO
from TrafficEnv import TrafficEnv
import numpy as np

def main():
    # Load the trained model
    # model_path = "ppo_traffic_model_final.zip"  # Adjust the path if necessary
    model_path = "models/ppo_traffic_model_3000_steps.zip"  # Adjust the path if necessary
    model = PPO.load(model_path)

    # Create the environment
    env = TrafficEnv()

    # Number of episodes to run
    num_episodes = 5

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            # Use the model to predict the action
            action, _states = model.predict(obs, deterministic=True)
            # Take the action in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            # Check if the episode is done
            done = terminated or truncated

            # Optional: Print step information
            # print(f"Episode: {episode + 1}, Step: {step_count}, Reward: {reward}")

        print(f"Episode {episode + 1} finished. Total Reward: {total_reward}")

    # Close the environment
    env.close()

if __name__ == '__main__':
    main()
