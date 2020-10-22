from chainerrl import wrappers
import matplotlib.pyplot as plt
from example_utils import train_chainer, make_chainer_dqn
from dacbench.benchmarks import FastDownwardBenchmark
from dacbench.wrappers import StateTrackingWrapper

bench = FastDownwardBenchmark()
env = bench.get_benchmark_env()
env = StateTrackingWrapper(env, 5)

# Chainer requires casting
env = wrappers.CastObservationToFloat32(env)

obs_size = env.observation_space.low.size
agent = make_chainer_dqn(obs_size, env.action_space)

train_chainer(agent, env)

img = env.render_state_tracking()
plt.show()
