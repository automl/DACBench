import chainer
from chainer import optimizers
from chainerrl import q_functions, wrappers, replay_buffer, explorers
from chainerrl.agents import DQN
import numpy as np
import gym

from daclib.wrappers import RewardNoiseWrapper


class DummyEnv(gym.Env):
    def __init__(self):
        self.c_step = None
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(1)
        self.reward_range = (-10, 10)

    def step(self, action):
        self.c_step += 1
        return np.array([0]), 0, self.c_step > 9, {}

    def reset(self):
        self.c_step = 0
        return np.array([1])


def train(env):
    num_episodes = 10
    for i in range(num_episodes):
        state = env.reset()
        done = False
        r = 0
        reward = 0
        while not done:
            action = agent.act_and_train(state, reward)
            next_state, reward, done, _ = env.step(action)
            r += reward
            state = next_state
        agent.stop_episode_and_train(state, reward, done=done)
        print(
            f"Episode {i}/{num_episodes}...........................................Reward: {r}"
        )


# We use a constant reward of 1 to demontrate the different noise values
env = DummyEnv()
# Chainer requires casting
env = wrappers.CastObservationToFloat32(env)

obs_space = env.observation_space
obs_size = obs_space.n
action_space = env.action_space
n_actions = action_space.n

q_func = q_functions.FCStateQFunctionWithDiscreteAction(obs_size, n_actions, 50, 1)
explorer = explorers.ConstantEpsilonGreedy(0.1, action_space.sample)
opt = optimizers.Adam(eps=1e-2)
opt.setup(q_func)
rbuf = replay_buffer.ReplayBuffer(10 ** 5)
agent = DQN(q_func, opt, rbuf, explorer=explorer, gamma=0.9)

print(
    "Demonstrating the most common distributions: standard versions of normal and exponential"
)
print("\n")
for noise_dist in ["standard_normal", "standard_exponential"]:
    print(f"Current noise distribution: {noise_dist}")
    print("Base reward is 0")
    wrapped = RewardNoiseWrapper(env, noise_dist=noise_dist)
    train(wrapped)
    print("\n")

print("Other distributions with added arguments")
print("\n")
for noise_dist, args in zip(
    ["normal", "uniform", "logistic"], [[0, 0.1], [-1, 1], [0, 2]]
):
    print(f"Current noise distribution: {noise_dist}")
    print("Base reward is 0")
    wrapped = RewardNoiseWrapper(env, noise_dist=noise_dist, dist_args=args)
    train(wrapped)
    print("\n")

print("Custom 'noise' function: always add 1")
print("\n")


def noise():
    return 1


wrapped = RewardNoiseWrapper(env, noise_function=noise)
train(wrapped)
