from chainerrl import wrappers
import matplotlib.pyplot as plt
from examples.example_utils import train_chainer, make_chainer_dqn
from dacbench.benchmarks import FastDownwardBenchmark
from dacbench.wrappers import StateTrackingWrapper

# Get FastDownward Environment
bench = FastDownwardBenchmark()
env = bench.get_environment()

# Wrap environment to track state
# In this case we also want the mean of each 5 step interval
env = StateTrackingWrapper(env, 5)

# Chainer requires casting to float32
env = wrappers.CastObservationToFloat32(env)

# Make chainer agent
obs_size = env.observation_space.low.size
agent = make_chainer_dqn(obs_size, env.action_space)

# Train for 10 episodes
train_chainer(agent, env)

# Plot state values after training
env.render_state_tracking()
plt.show()
