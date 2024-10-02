from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from TrafficEnv import TrafficEnv
from stable_baselines3.common.callbacks import CheckpointCallback

def main():
    env = TrafficEnv()

    # Optional: Check that the environment follows the Gymnasium API
    check_env(env)

    # Create the PPO model
    model = PPO('MlpPolicy', env, verbose=1)

    # Create a callback to save the model periodically
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./models/',
                                             name_prefix='ppo_traffic_model')

    # Train the model
    model.learn(total_timesteps=10000, callback=checkpoint_callback)

    # Save the final model
    model.save("ppo_traffic_model_final")

    # Close the environment
    env.close()

if __name__ == '__main__':
    main()
