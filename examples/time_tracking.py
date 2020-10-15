import chainer
from chainer import optimizers
from chainerrl import links, policies
from chainerrl.agents import a3c
import numpy as np
from daclib.benchmarks import LubyBenchmark
from daclib.wrappers import EpisodeTimeWrapper
from matplotlib import pyplot as plt


# Example model class taken from chainerrl examples:
# https://github.com/chainer/chainerrl/blob/master/examples/gym/train_a3c_gym.py
class A3CFFSoftmax(chainer.ChainList, a3c.A3CModel):
    """An example of A3C feedforward softmax policy."""

    def __init__(self, ndim_obs, n_actions, hidden_sizes=(200, 200)):
        self.pi = policies.SoftmaxPolicy(
            model=links.MLP(ndim_obs, n_actions, hidden_sizes)
        )
        self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


# We use the configuration from the "Learning to Optimize Step-size Adaption in CMA-ES" Paper by Shala et al.
bench = LubyBenchmark()
env = bench.get_benchmark()
env = EpisodeTimeWrapper(env, tracking_interval=10)

obs_space = env.observation_space
obs_size = obs_space.low.size
action_space = env.action_space
action_size = action_space.n

model = A3CFFSoftmax(obs_size, 1)
opt = optimizers.Adam(eps=1e-2)
opt.setup(model)
agent = a3c.A3C(model, opt, 10 ** 5, 0.9)

f, axarr = plt.subplots(2)
plt.axis("off")
plt.set_cmap("hot")
num_episodes = 10 ** 5
for i in range(num_episodes):
    state = env.reset()
    # Casting is necessary for chainerrl
    state = state.astype(np.float32)
    done = False
    r = 0
    reward = 0
    while not done:
        action = agent.act_and_train(state, reward)
        next_state, reward, done, _ = env.step(action)
        r += reward
        img = env.render_step_time()
        axarr[0].imshow(img)
        plt.pause(0.001)
        state = next_state.astype(np.float32)
    agent.stop_episode_and_train(state, reward, done=done)
    img = env.render_episode_time()
    axarr[1].imshow(img)
    plt.pause(0.001)
    print(
        f"Episode {i}/{num_episodes}...........................................Reward: {r}"
    )
