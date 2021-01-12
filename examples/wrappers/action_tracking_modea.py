from pathlib import Path

from dacbench.agents import RandomAgent
from dacbench.logger import Logger
from dacbench.runner import run_benchmark
from dacbench.benchmarks import ModeaBenchmark
from dacbench.wrappers import ActionFrequencyWrapper


# Make ModeaBenchmark environment
bench = ModeaBenchmark()
env = bench.get_environment()

logger = Logger(
    experiment_name=type(bench).__name__, output_path=Path("../plotting/data")
)
logger.set_env(env)
logger.add_benchmark(bench)
# Wrap environment to track action frequency
# In this case we also want the mean of each 5 step interval

env = ActionFrequencyWrapper(env, logger=logger.add_module(ActionFrequencyWrapper))

agent = RandomAgent(env)

run_benchmark(env, agent, 5, logger=logger)
