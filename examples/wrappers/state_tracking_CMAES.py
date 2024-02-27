"""Example for the state tracking using CMAES."""
from pathlib import Path

from dacbench.agents import RandomAgent
from dacbench.benchmarks import CMAESBenchmark
from dacbench.logger import Logger
from dacbench.runner import run_benchmark
from dacbench.wrappers import StateTrackingWrapper

# Make CMAESBenchmark environment
bench = CMAESBenchmark()
env = bench.get_environment()

# Make Logger object to track state information
logger = Logger(
    experiment_name=type(bench).__name__, output_path=Path("../plotting/data")
)
logger.set_env(env)

# Wrap env with StateTrackingWrapper
env = StateTrackingWrapper(env, logger=logger.add_module(StateTrackingWrapper))

# Run random agent for 5 episodes and log state information to file
# You can plot these results with the plotting examples
agent = RandomAgent(env)
run_benchmark(env, agent, 5, logger=logger)
logger.close()
