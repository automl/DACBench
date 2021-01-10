import json
from pathlib import Path
import seaborn as sns
from dacbench.logger import Logger, log2dataframe
from dacbench.agents.simple_agents import RandomAgent
from dacbench.benchmarks import SigmoidBenchmark
from dacbench.runner import run_benchmark
from dacbench.wrappers import PerformanceTrackingWrapper

import matplotlib.pyplot as plt


if __name__ == "__main__":

    bench = SigmoidBenchmark()
    num_episodes = 10
    seeds = range(10)

    logger = Logger(
        experiment_name="sigmoid_example",
        output_path=Path("output"),
        step_write_frequency=None,
        episode_write_frequency=None,
    )
    performance_logger = logger.add_module(PerformanceTrackingWrapper)

    for s in seeds:
        logger.set_additional_info(seed=s)
        env = bench.get_benchmark(seed=s)
        env = PerformanceTrackingWrapper(env, logger=performance_logger)
        logger.set_env(env)
        agent = RandomAgent(env)
        run_benchmark(env, agent, num_episodes, logger)

    logger.close()

    with open(performance_logger.get_logfile(), "r") as f:
        logs = list(map(json.loads, f))

    dataframe = log2dataframe(logs, wide=True)
    sns.relplot(
        data=dataframe,
        x="episode",
        y="overall_performance",
        kind="line",
        row="instance",
    )
    plt.show()
