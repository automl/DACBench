from chainer import optimizers
from chainerrl import q_functions, wrappers, replay_buffer, explorers
from chainerrl.agents import DQN
import matplotlib.pyplot as plt
import numpy as np
import gym

from daclib.benchmarks import LubyBenchmark
from daclib.wrappers import ActionFrequencyWrapper

def train(env):
    num_episodes = 20
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


bench = LubyBenchmark()
env = bench.get_benchmark_env()
env = ActionFrequencyWrapper(env, 5)
# Chainer requires casting
env = wrappers.CastObservationToFloat32(env)

obs_space = env.observation_space
obs_size = obs_space.low.size
action_space = env.action_space
n_actions = action_space.n

q_func = q_functions.FCStateQFunctionWithDiscreteAction(obs_size, n_actions, 50, 1)
explorer = explorers.ConstantEpsilonGreedy(0.1, action_space.sample)
opt = optimizers.Adam(eps=1e-2)
opt.setup(q_func)
rbuf = replay_buffer.ReplayBuffer(10 ** 5)
agent = DQN(q_func, opt, rbuf, explorer=explorer, gamma=0.9)

train(env)
img = env.render_action_tracking()
plt.axis("off")
plt.imshow(img)
plt.show()
