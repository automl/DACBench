import os
import json
import numpy as np
import matplotlib.pyplot as plt
from dacbench import benchmarks
from dacbench.wrappers import PerformanceTrackingWrapper
import seaborn as sb

sb.set_style("darkgrid")
current_palette = list(sb.color_palette())


def run_benchmark(env, agent, num_episodes):
    """
    Run single benchmark env for a given number of episodes with a given agent

    Parameters
    -------
    env : gym.Env
        Benchmark environment
    agent
        Any agent implementing the methods act, train and end_episode (see AbstractDACBenchAgent below)
    num_episodes : int
        Number of episodes to run
    """
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        reward = 0
        while not done:
            action = agent.act(state, reward)
            next_state, reward, done, _ = env.step(action)
            agent.train(next_state, reward)
            state = next_state
        agent.end_episode(state, reward)


def run_dacbench(results_path, agent_method, num_episodes):
    """
    Run all benchmarks for 10 seeds for a given number of episodes with a given agent and save result

    Parameters
    -------
    results_path : str
        Path to where results should be saved
    agent_method : function
        Method that takes an env as input and returns an agent
    num_episodes : int
        Number of episodes to run for each benchmark
    """
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    for b in map(benchmarks.__dict__.get, benchmarks.__all__):
        overall = []
        print(f"Evaluating {b.__name__}")
        for i in range(10):
            print(f"Seed {i}/10")
            bench = b()
            env = bench.get_benchmark(seed=i)
            env = PerformanceTrackingWrapper(env)
            agent = agent_method(env)
            run_benchmark(env, agent, num_episodes)
            performance = env.get_performance()[0]
            overall.append(performance)
        print("\n")
        file_name = results_path + "/" + b.__name__ + ".json"
        with open(file_name, "w+") as fp:
            json.dump(overall, fp)


def plot_results(path):
    """
    Load and plot results from file

    Parameters
    -------
    path : str
        Path to result directory
    """
    performances = {}
    stds = {}
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.endswith(".json"):
                key = entry.name.split(".")[0]
                filename = path + "/" + entry.name
                with open(filename, "r") as fp:
                    overall = json.load(fp)
                    performances[key] = np.mean(overall, axis=0)
                    stds[key] = np.std(overall, axis=0)

    num_benchmarks = len(list(performances.keys()))
    if num_benchmarks > 5:
        xs = num_benchmarks // 5
        ys = num_benchmarks % 5
        figure, axs = plt.subplots(xs, ys, figsize=(12, 12))
    else:
        figure, axs = plt.subplots(num_benchmarks, figsize=(12, 12))
    plt.tight_layout()
    for k, i in zip(performances.keys(), np.arange(num_benchmarks)):
        plt.subplots_adjust(hspace=0.4)
        perf = np.array(performances[k])
        perf = np.interp(perf, (perf.min(), perf.max()), (-1, +1))
        std = np.interp(stds[k], (perf.min(), perf.max()), (-1, +1))
        if num_benchmarks > 5:
            axs[i // 5, i % 5].set_xlabel("Episodes")
            axs[i // 5, i % 5].set_ylabel("Reward")
            axs[i // 5, i % 5].set_title(k)
            axs[i // 5, i % 5].plot(np.arange(len(perf)), perf, label=k)
            axs[i // 5, i % 5].fill_between(
                np.arange(len(perf)), perf - std, perf + std, alpha=0.25
            )
        else:
            axs[i].set_xlabel("Episodes")
            axs[i].set_ylabel("Reward")
            axs[i].set_title(k)
            axs[i].plot(np.arange(len(perf)), perf, label=k)
            axs[i].fill_between(
                np.arange(len(perf)), perf - std, perf + std, alpha=0.25
            )
    plt.show()


class AbstractDACBenchAgent:
    """ Abstract class to implement for use with the runner function """
    def __init__(self, env):
        """
        Initialize agent

        Parameters
        -------
        env : gym.Env
            Environment to train on
        """
        pass

    def act(self, state, reward):
        """
        Compute and return environment action

        Parameters
        -------
        state
            Environment state
        reward
            Environment reward

        Returns
        -------
        action
            Action to take
        """
        raise NotImplementedError

    def train(self, next_state, reward):
        """
        Train during episode if needed (pass if not)

        Parameters
        -------
        next_state
            Environment state after step
        reward
            Environment reward
        """
        raise NotImplementedError

    def end_episode(self, state, reward):
        """
        End of episode training if needed (pass if not)

        Parameters
        -------
        state
            Environment state
        reward
            Environment reward
        """
        raise NotImplementedError
