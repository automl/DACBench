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
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    for b in map(benchmarks.__dict__.get, benchmarks.__all__):
        bench = b()
        env = bench.get_benchmark()
        env = PerformanceTrackingWrapper(env)
        agent = agent_method(env)
        run_benchmark(env, agent, num_episodes)
        performance = env.get_performance()[0]
        file_name = results_path + "/" + b.__name__ + ".json"
        with open(file_name, "w+") as fp:
            json.dump(performance, fp)


def plot_results(path):
    performances = {}
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.endswith(".json"):
                key = entry.name.split(".")[0]
                filename = path + "/" + entry.name
                with open(filename, "r") as fp:
                    performances[key] = json.load(fp)

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
        if num_benchmarks > 5:
            axs[i // 5, i % 5].set_xlabel("Episodes")
            axs[i // 5, i % 5].set_ylabel("Reward")
            axs[i // 5, i % 5].set_title(k)
            axs[i // 5, i % 5].plot(np.arange(len(perf)), perf, label=k)
        else:
            axs[i].set_xlabel("Episodes")
            axs[i].set_ylabel("Reward")
            axs[i].set_title(k)
            axs[i].plot(np.arange(len(perf)), perf, label=k)
    plt.show()


class AbstractDACBenchAgent:
    def __init__(self, env):
        raise NotImplementedError

    def act(self, state, reward):
        raise NotImplementedError

    def train(self, next_state, reward):
        raise NotImplementedError

    def end_episode(self, state, reward):
        raise NotImplementedError
