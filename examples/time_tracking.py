import matplotlib.pyplot as plt
import numpy as np
from example_utils import make_chainer_a3c, train_chainer
from dacbench.benchmarks import CMAESBenchmark
from dacbench.wrappers import EpisodeTimeWrapper


# Helper function to flatten observation space
def flatten(li):
    return [value for sublist in li for value in sublist]


# Make CMA-ES environment
# We use the configuration from the "Learning to Optimize Step-size Adaption in CMA-ES" Paper by Shala et al.
bench = CMAESBenchmark()
env = bench.get_benchmark()

# Wrap environment to track time
# Here we also want the mean of each 2 step interval
env = EpisodeTimeWrapper(env, 2)

# Make chainer agent
space_array = [
    env.observation_space[k].low for k in list(env.observation_space.spaces.keys())
]
obs_size = np.array(flatten(space_array)).size
action_size = env.action_space.low.size
agent = make_chainer_a3c(obs_size, action_size)

# Train agent for 10 episodes
train_chainer(agent, env, flatten_state=True)

# After training:
# Plot time spent per step
env.render_step_time()
plt.show()
# Plot time spent per episode
env.render_episode_time()
plt.show()
