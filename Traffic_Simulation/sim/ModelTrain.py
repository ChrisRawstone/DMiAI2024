from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from TrafficEnv import TrafficEnv
import gymnasium as gym  # Import gymnasium
import os
from stable_baselines3.common.env_checker import check_env


class StatsLoggingCallback(BaseCallback):
    def __init__(self, eval_callback, log_dir, verbose=1):
        super(StatsLoggingCallback, self).__init__(verbose)
        self.eval_callback = eval_callback
        self.log_dir = log_dir
        self.best_mean_reward = -float('inf')
        self.filepath = os.path.join(self.log_dir, "training_stats.txt")

    def _on_step(self):
        if self.eval_callback.best_mean_reward > self.best_mean_reward:
            self.best_mean_reward = self.eval_callback.best_mean_reward
            with open(self.filepath, "a") as f:
                f.write(f"New best model at step {self.num_timesteps}:\n")
                f.write(f"Mean Reward: {self.best_mean_reward}\n")
                f.write(f"----------------------------------------\n")
        return True

def main():
    # Create and check the environment
    env = TrafficEnv()
    check_env(env)  # Use the unwrapped environment for checking

    # Wrap the environment for training
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Directory to save the models and log files
    log_dir = "./models/"
    saved_model_path = os.path.join(log_dir, "ppo_traffic_model_latest.zip")

    if os.path.exists(saved_model_path):
        print(f"Loading saved model from {saved_model_path} to continue training.")
        model = PPO.load(saved_model_path, env=env)
    else:
        print("No saved model found. Creating a new model.")
        model = PPO(
            'MlpPolicy',
            env,
            verbose=1,
            learning_rate=0.0001,
            n_steps=256,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5
        )

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix='ppo_traffic_model')
    eval_env = DummyVecEnv([lambda: TrafficEnv()])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    stats_logging_callback = StatsLoggingCallback(eval_callback, log_dir)
    total_timesteps = 100000
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback, stats_logging_callback])

    model.save(os.path.join(log_dir, "ppo_traffic_model_final"))
    env.save(os.path.join(log_dir, "vec_normalize.pkl"))
    env.close()

if __name__ == '__main__':
    main()
