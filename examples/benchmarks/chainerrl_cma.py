from pathlib import Path
from dacbench.logger import Logger
from dacbench.wrappers import PerformanceTrackingWrapper, ObservationWrapper
from examples.example_utils import make_chainer_a3c
from dacbench.benchmarks import CMAESBenchmark

# Make logger object
logger = Logger(experiment_name="CMAESBenchmark", output_path=Path("../plotting/data"))

# Make CMA-ES environment
# We use the configuration from the "Learning to Optimize Step-size Adaption in CMA-ES" Paper by Shala et al.
bench = CMAESBenchmark()
env = bench.get_benchmark()
logger.set_env(env)

# Wrap to track performance
performance_logger = logger.add_module(PerformanceTrackingWrapper)
env = PerformanceTrackingWrapper(env=env, logger=performance_logger)

# Also wrap to make the dictionary observations into an easy to work with list
env = ObservationWrapper(env)

# Make chainer agent
obs_size = env.observation_space.low.size
action_size = env.action_space.low.size
agent = make_chainer_a3c(obs_size, action_size)

# Training
num_episodes = 3
for i in range(num_episodes):
    # Reset environment to begin episode
    state = env.reset()

    # Initialize episode
    done = False
    r = 0
    reward = 0
    while not done:
        # Select action
        action = agent.act_and_train(state, reward)
        # Execute action
        next_state, reward, done, _ = env.step(action)
        r += reward
        logger.next_step()
        state = next_state
    logger.next_episode()
    # Train agent after episode has ended
    agent.stop_episode_and_train(state, reward, done=done)
    # Log episode
    print(
        f"Episode {i+1}/{num_episodes}...........................................Reward: {r}"
    )
