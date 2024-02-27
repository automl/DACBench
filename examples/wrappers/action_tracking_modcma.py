"""Example for the action tracking wrapper showing on modcma."""
from pathlib import Path

from dacbench.agents import RandomAgent
from dacbench.benchmarks import ModCMABenchmark
from dacbench.logger import Logger
from dacbench.runner import run_benchmark
from dacbench.wrappers import ActionFrequencyWrapper

# Make ModeaBenchmark environment
bench = ModCMABenchmark()
env = bench.get_environment()

# Make logger object
logger = Logger(
    experiment_name=type(bench).__name__, output_path=Path("../plotting/data")
)
logger.set_env(env)
logger.add_benchmark(bench)

# Wrap environment to track action frequency
env = ActionFrequencyWrapper(env, logger=logger.add_module(ActionFrequencyWrapper))

# Run random agent for 5 episodes and log actions to file
agent = RandomAgent(env)
run_benchmark(env, agent, 5, logger=logger)
