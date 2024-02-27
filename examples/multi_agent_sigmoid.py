"""Example for the multi agent with sigmoid."""
from dacbench.benchmarks import SigmoidBenchmark

bench = SigmoidBenchmark()
bench.config["multi_agent"] = True
env = bench.get_environment()
env.register_agent(0)
env.register_agent(1)
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
