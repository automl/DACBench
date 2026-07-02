"""Run and log an experiment."""

from pathlib import Path

import matplotlib.pyplot as plt
from dacbench.agents.simple_agents import RandomAgent
from dacbench.benchmarks import FunctionApproximationBenchmark
from dacbench.logger import Logger, load_logs, log2dataframe
from dacbench.plotting import plot_performance, plot_performance_per_instance
from dacbench.runner import run_benchmark
from dacbench.wrappers import PerformanceTrackingWrapper, StateTrackingWrapper

# Run an experiment and log the results
if __name__ == "__main__":
    # Make benchmark
    bench = FunctionApproximationBenchmark()

    # Run for 10 episodes each on 10 seeds
    num_episodes = 10
    seeds = range(10)

    # Make logger object and add modules for performance & state logging
    logger = Logger(
        experiment_name="function_approximation_example",
        output_path=Path("plotting/data"),
        step_write_frequency=None,
        episode_write_frequency=None,
    )
    state_logger = logger.add_module(StateTrackingWrapper)
    performance_logger = logger.add_module(PerformanceTrackingWrapper)

    for s in seeds:
        # Make & wrap benchmark environment
        env = bench.get_benchmark(seed=s)
        env = PerformanceTrackingWrapper(env, logger=performance_logger)
        env = StateTrackingWrapper(env, logger=state_logger)

        # Add env to logger
        logger.set_env(env)

        # Run random agent
        agent = RandomAgent(env)
        run_benchmark(env, agent, num_episodes, logger)

    # Close logger object
    logger.close()

    # Load performance of last seed into pandas DataFrame
    logs = load_logs(performance_logger.get_logfile())
    dataframe = log2dataframe(logs, wide=True)

    # Plot overall performance
    plot_performance(dataframe)
    plt.show()

    # Plot performance per instance
    plot_performance_per_instance(dataframe)
    plt.show()
