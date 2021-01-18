import tensorflow as tf
import warnings
from stable_baselines import PPO2
from dacbench.benchmarks import SigmoidBenchmark
from dacbench.wrappers import PerformanceTrackingWrapper

# Baselines uses an old TF version
# As soon as baselines3 is out of beta, we should use it
# Until then we try to surpress as many of these warnings as possible
warnings.simplefilter(action="ignore", category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)


# Environment creation method
# Overwrite standard configuration with given options, but adapt action space automatically
def make_sigmoid(config):
    bench = SigmoidBenchmark()
    for k in config.keys():
        if k == "action_values":
            bench.set_action_values(config[k])
        else:
            bench.config[k] = config[k]
    return bench.get_environment()


# Experiment configuration
# Play 5D scenario
action_values = (3, 3, 3, 3, 3)
env_config = {
    "seed": 0,
    "action_values": action_values,
    "instance_set_path": "../instance_sets/sigmoid/sigmoid_5D3M_train.csv",
}

# Make environment
# To track rewards we use our wrapper (this is only for simplicity)
env = make_sigmoid(env_config)
env = PerformanceTrackingWrapper(env)

# Make simple PPO policy
model = PPO2("MlpPolicy", env)

# Run for 10 steps
model.learn(total_timesteps=200)

performance = env.get_performance()[0]
for i in range(len(performance)):
    print(
        f"Episode {i+1}/{len(performance)}...........................................Reward: {performance[i]}"
    )
