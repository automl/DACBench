import chainer
from chainer import optimizers
from chainerrl import links, policies
from chainerrl.agents import a3c
import numpy as np
from daclib.benchmarks import CMAESBenchmark


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


def flatten(li):
    return [value for sublist in li for value in sublist]


# We use the configuration from the "Learning to Optimize Step-size Adaption in CMA-ES" Paper by Shala et al.
bench = CMAESBenchmark()
env = bench.get_complete_benchmark()

obs_space = env.observation_space
space_array = [obs_space[k].low for k in list(obs_space.spaces.keys())]
obs_size = np.array(flatten(space_array)).size
action_space = env.action_space
action_size = action_space.low.size

model = A3CFFSoftmax(obs_size, action_size)
opt = optimizers.Adam(eps=1e-2)
opt.setup(model)
agent = a3c.A3C(model, opt, 10 ** 5, 0.9)

num_episodes = 10 ** 5
for i in range(num_episodes):
    state = env.reset()
    # Flattening state
    state = np.array(flatten([state[k] for k in state.keys()]))
    # Casting is necessary for chainerrl
    state = state.astype(np.float32)
    done = False
    r = 0
    reward = 0
    while not done:
        action = agent.act_and_train(state, reward)
        next_state, reward, done, _ = env.step(action)
        r += reward
        state = np.array(flatten([next_state[k] for k in next_state.keys()]))
        state = state.astype(np.float32)
    agent.stop_episode_and_train(state, reward, done=done)
    print(
        f"Episode {i}/{num_episodes}...........................................Reward: {r}"
    )
