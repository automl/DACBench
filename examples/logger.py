import json
from pathlib import Path
import seaborn as sns
from dacbench.logger import Logger, log2dataframe
from dacbench.agents.simple_agents import RandomAgent
from dacbench.benchmarks import SigmoidBenchmark
from dacbench.wrappers import PerformanceTrackingWrapper

import matplotlib.pyplot as plt


def run_benchmark(env, agent, num_episodes, logger):

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        reward = 0
        while not done:
            action = agent.act(state, reward)
            next_state, reward, done, _ = env.step(action)
            agent.train(next_state, reward)
            state = next_state
            logger.next_step()
        agent.end_episode(state, reward)
        logger.next_episode()
    env.close()


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
        x="step",
        y="episode_performance",
        hue="episode",
        kind="line",
        col="seed",
        row="instance",
    )
    plt.show()
