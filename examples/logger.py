from pathlib import Path

from dacbench.plotting import plot_performance, plot_performance_per_instance
from dacbench.logger import Logger, log2dataframe, load_logs
from dacbench.agents.simple_agents import RandomAgent
from dacbench.benchmarks import SigmoidBenchmark
from dacbench.runner import run_benchmark
from dacbench.wrappers import PerformanceTrackingWrapper, StateTrackingWrapper

import matplotlib.pyplot as plt


if __name__ == "__main__":

    bench = SigmoidBenchmark()
    num_episodes = 10
    seeds = range(10)

    logger = Logger(
        experiment_name="sigmoid_example",
        output_path=Path("plotting/data"),
        step_write_frequency=None,
        episode_write_frequency=None,
    )
    state_logger = logger.add_module(StateTrackingWrapper)
    performance_logger = logger.add_module(PerformanceTrackingWrapper)

    for s in seeds:
        logger.set_additional_info(seed=s)
        env = bench.get_benchmark(seed=s)
        env = PerformanceTrackingWrapper(env, logger=performance_logger)
        env = StateTrackingWrapper(env, logger=state_logger)
        logger.set_env(env)
        agent = RandomAgent(env)
        run_benchmark(env, agent, num_episodes, logger)

    logger.close()

    logs = load_logs(performance_logger.get_logfile())
    dataframe = log2dataframe(logs, wide=True)
    plot_performance(dataframe)
    plt.show()

    plot_performance_per_instance(dataframe)
    plt.show()
