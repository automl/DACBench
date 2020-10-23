import matplotlib.pyplot as plt
from chainerrl import wrappers
from example_utils import train_chainer, make_chainer_dqn
from dacbench.benchmarks import LubyBenchmark
from dacbench.wrappers import ActionFrequencyWrapper


# Make Luby environment
bench = LubyBenchmark()
env = bench.get_environment()

# Wrap environment to track action frequency
# In this case we also want the mean of each 5 step interval
env = ActionFrequencyWrapper(env, 5)

# Chainer requires casting to float32
env = wrappers.CastObservationToFloat32(env)

# Make chainer agent
obs_size = env.observation_space.low.size
agent = make_chainer_dqn(obs_size, env.action_space)

# Train agent for 10 episodes
train_chainer(agent, env)

# Plot action progression after training
env.render_action_tracking()
plt.show()
