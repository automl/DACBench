from chainer import optimizers
from chainerrl import q_functions, wrappers, replay_buffer, explorers
from chainerrl.agents import DQN
import numpy as np
import gym

from example_utils import DummyEnv, train_chainer, make_chainer_dqn
from daclib.wrappers import RewardNoiseWrapper

# We use a constant reward of 1 to demontrate the different noise values
env = DummyEnv()
# Chainer requires casting
env = wrappers.CastObservationToFloat32(env)

obs_size = env.observation_space.n
make_chainer_dqn(obs_size, env.action_space)

print(
    "Demonstrating the most common distributions: standard versions of normal and exponential"
)
print("\n")
for noise_dist in ["standard_normal", "standard_exponential"]:
    print(f"Current noise distribution: {noise_dist}")
    print("Base reward is 0")
    wrapped = RewardNoiseWrapper(env, noise_dist=noise_dist)
    train_chainer(wrapped)
    print("\n")

print("Other distributions with added arguments")
print("\n")
for noise_dist, args in zip(
    ["normal", "uniform", "logistic"], [[0, 0.1], [-1, 1], [0, 2]]
):
    print(f"Current noise distribution: {noise_dist}")
    print("Base reward is 0")
    wrapped = RewardNoiseWrapper(env, noise_dist=noise_dist, dist_args=args)
    train_chainer(wrapped)
    print("\n")

print("Custom 'noise' function: always add 1")
print("\n")

def noise():
    return 1

wrapped = RewardNoiseWrapper(env, noise_function=noise)
train_chainer(wrapped)
