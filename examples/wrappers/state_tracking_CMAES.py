from pathlib import Path

from dacbench.agents import RandomAgent
from dacbench.logger import Logger
from dacbench.runner import run_benchmark
from dacbench.benchmarks import CMAESBenchmark
from dacbench.wrappers import StateTrackingWrapper

# Make CMAESBenchmark environment
bench = CMAESBenchmark()
env = bench.get_environment()

logger = Logger(
    experiment_name=type(bench).__name__, output_path=Path("../plotting/data")
)
logger.set_env(env)

env = StateTrackingWrapper(env, logger=logger.add_module(StateTrackingWrapper))

agent = RandomAgent(env)

run_benchmark(env, agent, 5, logger=logger)
