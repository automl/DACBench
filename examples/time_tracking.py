import chainer
from chainer import optimizers
from chainerrl import links, policies
from chainerrl.agents import a3c
import matplotlib.pyplot as plt
import numpy as np

from example_utils import make_chainer_a3c, train_chainer
from dacbench.benchmarks import CMAESBenchmark
from dacbench.wrappers import EpisodeTimeWrapper

def flatten(li):                                                                                                            return [value for sublist in li for value in sublist]  

# We use the configuration from the "Learning to Optimize Step-size Adaption in CMA-ES" Paper by Shala et al.
bench = CMAESBenchmark()
env = bench.get_benchmark()
env = EpisodeTimeWrapper(env, 2)

obs_space = env.observation_space
space_array = [obs_space[k].low for k in list(obs_space.spaces.keys())]
obs_size = np.array(flatten(space_array)).size
action_space = env.action_space
action_size = action_space.low.size

agent = make_chainer_a3c(obs_size, action_size)

train_chainer(agent, env, flatten_state=True)

img = env.render_step_time()
plt.show()
img = env.render_episode_time()
plt.show()
