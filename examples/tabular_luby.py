"""
Code adapted from
"Dynamic Algorithm Configuration:Foundation of a New Meta-Algorithmic Framework"
by A. Biedenkapp and H. F. Bozkurt and T. Eimer and F. Hutter and M. Lindauer.
Original environment authors: André Biedenkapp, H. Furkan Bozkurt
"""

from example_utils import q_learning
from dacbench.benchmarks import LubyBenchmark

# Make Luby environment
bench = LubyBenchmark()
env = bench.get_benchmark_env()

# Execute 10 episodes of tabular Q-Learning
q_func, test_train_stats = q_learning(env, 10)
print(f"Rewards: {test_train_stats[1].episode_rewards}")
print(f"Episode Lenghts: {test_train_stats[1].episode_lengths}")
