from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from TrafficEnv import TrafficEnv
import os

# Custom callback to save training stats to a txt file when the best model is saved
class StatsLoggingCallback(BaseCallback):
    def __init__(self, eval_callback, log_dir, verbose=1):
        super(StatsLoggingCallback, self).__init__(verbose)
        self.eval_callback = eval_callback
        self.log_dir = log_dir
        self.best_mean_reward = -float('inf')
        self.filepath = os.path.join(self.log_dir, "training_stats.txt")

    def _on_step(self):
        # Check if a new best model was found
        if self.eval_callback.best_mean_reward > self.best_mean_reward:
            self.best_mean_reward = self.eval_callback.best_mean_reward
            # Save the stats in the txt file
            with open(self.filepath, "a") as f:
                f.write(f"New best model at step {self.num_timesteps}:\n")
                f.write(f"Mean Reward: {self.best_mean_reward}\n")
                f.write(f"----------------------------------------\n")
        return True

def main():
    env = TrafficEnv()

    # Optional: Check that the environment follows the Gymnasium API
    check_env(env)

    # Directory to save the models and log files
    log_dir = "./models/"

    # Path to the saved model (if any)
    saved_model_path = os.path.join(log_dir, "ppo_traffic_model_final.zip")

    # Check if a saved model exists
    if os.path.exists(saved_model_path):
        print(f"Loading saved model from {saved_model_path} to continue training.")
        # Load the saved model
        model = PPO.load(saved_model_path, env=env)
    else:
        print("No saved model found. Creating a new model.")
        # Create a new PPO model from scratch with custom hyperparameters
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=0.0001,   # Slow, stable learning
        n_steps=300,            # Match with episode length (300 steps)
        batch_size=64,          # Moderate batch size for smoother updates
        n_epochs=10,            # More epochs to refine updates
        gamma=0.99,             # Value long-term rewards
        gae_lambda=0.95,        # Generalized Advantage Estimation factor
        clip_range=0.2,         # Conservative clipping range for stability
        ent_coef=0.01,          # Encourage exploration (early in training)
        vf_coef=0.5,            # Value function importance
        max_grad_norm=0.5       # Prevent gradient explosion
    )


    # Create a callback to save the model periodically
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir,
                                             name_prefix='ppo_traffic_model')

    # Create an environment for evaluation
    eval_env = TrafficEnv()

    # Create an EvalCallback to save the best model based on performance
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=1000,  # Evaluate every 1000 steps
        deterministic=True,
        render=False
    )

    # Custom callback for logging stats when the best model is saved
    stats_logging_callback = StatsLoggingCallback(eval_callback, log_dir)

    # Train the model with the combined callbacks
    total_timesteps = 10000  # Adjust as needed
    model.learn(total_timesteps=total_timesteps, 
                callback=[checkpoint_callback, eval_callback, stats_logging_callback])

    # Save the final model
    model.save(os.path.join(log_dir, "ppo_traffic_model_final"))

    # Close the environment
    env.close()

if __name__ == '__main__':
    main()
