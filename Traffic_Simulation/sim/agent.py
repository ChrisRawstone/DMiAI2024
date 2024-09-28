from stable_baselines3 import PPO
from trafficenv import TrafficEnv
# Initialize the environment
env = TrafficEnv()


# Create and train the PPO model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)  # Adjust based on your needs

# Save the model
model.save("ppo_traffic_control")

# Test the trained model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
