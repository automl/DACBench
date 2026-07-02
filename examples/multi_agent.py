"""Example for the multi agent with sigmoid."""

from dacbench.benchmarks import FunctionApproximationBenchmark

bench = FunctionApproximationBenchmark()
bench.config["multi_agent"] = True
env = bench.get_environment()
env.register_agent("value_dim_1")
env.register_agent("value_dim_2")
env.reset()
terminated, truncated = False, False
total_reward = 0
while not (terminated or truncated):
    for a in [0, 1]:
        observation, reward, terminated, truncated, info = env.last()
        action = env.action_spaces[a].sample()
        env.step(action)
    observation, reward, terminated, truncated, info = env.last()
    total_reward += reward

print(f"The final reward was {total_reward}.")
