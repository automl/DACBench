import chainer
from chainer import optimizers
from chainerrl import links, policies
from chainerrl.agents import a3c
import numpy as np

from example_utils import make_chainer_a3c
from daclib.benchmarks import CMAESBenchmark

def flatten(li):
    return [value for sublist in li for value in sublist]

# We use the configuration from the "Learning to Optimize Step-size Adaption in CMA-ES" Paper by Shala et al.
bench = CMAESBenchmark()
env = bench.get_benchmark()

space_array = [env.observation_space[k].low for k in list(env.observation_space.spaces.keys())]
obs_size = np.array(flatten(space_array)).size
action_size = env.action_space.low.size

agent = make_chainer_a3c(obs_size, action_size)

num_episodes = 10
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
