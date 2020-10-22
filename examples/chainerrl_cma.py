import numpy as np
from example_utils import make_chainer_a3c
from dacbench.benchmarks import CMAESBenchmark

# Helper method to flatten observation space
def flatten(li):
    return [value for sublist in li for value in sublist]

# Make CMA-ES environment
# We use the configuration from the "Learning to Optimize Step-size Adaption in CMA-ES" Paper by Shala et al.
bench = CMAESBenchmark()
env = bench.get_benchmark()

# Make chainer agent
space_array = [
    env.observation_space[k].low for k in list(env.observation_space.spaces.keys())
]
obs_size = np.array(flatten(space_array)).size
action_size = env.action_space.low.size
agent = make_chainer_a3c(obs_size, action_size)

# Training
num_episodes = 10
for i in range(num_episodes):
    # Reset environment to begin episode
    state = env.reset()
    # Flattening state
    state = np.array(flatten([state[k] for k in state.keys()]))
    # Casting is necessary for chainer
    state = state.astype(np.float32)

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
        state = np.array(flatten([next_state[k] for k in next_state.keys()]))
        state = state.astype(np.float32)
    # Train agent after episode has ended
    agent.stop_episode_and_train(state, reward, done=done)
    # Log episode
    print(
        f"Episode {i}/{num_episodes}...........................................Reward: {r}"
    )
