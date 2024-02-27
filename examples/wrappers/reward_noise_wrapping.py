"""Example for using the reward noise wrapper."""
from chainerrl import wrappers
from dacbench.wrappers import RewardNoiseWrapper
from examples.example_utils import DummyEnv, make_chainer_dqn, train_chainer

# We use a dummy env with constant reward of 1 to demontrate the different noise values
env = DummyEnv()
# Chainer requires casting
env = wrappers.CastObservationToFloat32(env)

# Make chainer agent
obs_size = env.observation_space.n
agent = make_chainer_dqn(obs_size, env.action_space)

# First example: Adding reward noise from the default settings of normal and exponential distributions
print(
    "Demonstrating the most common distributions: standard versions of normal and exponential"
)
print("\n")
for noise_dist in ["standard_normal", "standard_exponential"]:
    print(f"Current noise distribution: {noise_dist}")
    print("Base reward is 0")
    wrapped = RewardNoiseWrapper(env, noise_dist=noise_dist)
    train_chainer(agent, wrapped)
    print("\n")

# Second example: Using customized reward noise distributions
print("Other distributions with added arguments")
print("\n")
for noise_dist, args in zip(
    ["normal", "uniform", "logistic"], [[0, 0.1], [-1, 1], [0, 2]], strict=False
):
    print(f"Current noise distribution: {noise_dist}")
    print("Base reward is 0")
    wrapped = RewardNoiseWrapper(env, noise_dist=noise_dist, dist_args=args)
    train_chainer(agent, wrapped)
    print("\n")

# Third example: using noise from a custom noise function
print("Custom 'noise' function: always add 1")
print("\n")


def noise():
    return 1


wrapped = RewardNoiseWrapper(env, noise_function=noise)
train_chainer(agent, wrapped)
