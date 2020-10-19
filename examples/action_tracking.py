import matplotlib.pyplot as plt
import numpy as np
import gym

from example_utils import train_chainer, make_chainer_dqn
from daclib.benchmarks import LubyBenchmark
from daclib.wrappers import ActionFrequencyWrapper


bench = LubyBenchmark()
env = bench.get_benchmark_env()
env = ActionFrequencyWrapper(env, 5)

# Chainer requires casting
env = wrappers.CastObservationToFloat32(env)

obs_size = env.observation_space.low.size
make_chainer_dqn(obs_size, env.action_space)

# Train agent in any way
train_chainer(env)
img = env.render_action_tracking()
plt.axis("off")
plt.imshow(img)
plt.show()
