"""Example for plotting of needed average time."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from dacbench.logger import load_logs, log2dataframe
from dacbench.plotting import plot_episode_time, plot_step_time


def step_time_example(data):
    """Plot time spent per step on average and split by seed

    Parameters
    ----------
    data : pd.DataFrame
        The non-wide data frame resulting from loading the logging results from EpisodeTimeTracker
    """
    grid = plot_step_time(data, y_label="Step Duration [s]")
    grid.savefig("output/sigmoid_step_duration.pdf")
    plt.show()

    grid = plot_step_time(data, y_label="Step Duration [s]", hue="seed")
    grid.savefig("output/sigmoid_step_duration_per_seed.pdf")
    plt.show()


def episode_time_example(data):
    """Plot time spent per episode

    Parameters
    ----------
    data : pd.DataFrame
        The non-wide data frame resulting from loading the logging results from EpisodeTimeTracker
    """
    print(data[~data.episode_duration.isna()])
    grid = plot_episode_time(
        data[~data.episode_duration.isna()], y_label="Episode Duration [s]"
    )
    grid.savefig("output/sigmoid_episode_duration.pdf")
    plt.show()


def step_time_interval_example(data: pd.DataFrame, interval: int = 10):
    """Plot mean time spent on steps in a given interval

    Parameters
    ----------
    data : pd.DataFrame
        The non-wide data frame resulting from loading the logging results from EpisodeTimeTracker
    interval : int
        Number of steps to average over
    """
    grid = plot_step_time(data, interval, title="Mean Step Duration")
    grid.savefig("output/sigmoid_step_duration.pdf")
    plt.show()


if __name__ == "__main__":
    # Load data from file into pandas DataFrame
    file = Path("data/sigmoid_example/EpisodeTimeWrapper.jsonl")
    logs = load_logs(file)
    data = log2dataframe(logs, wide=True, drop_columns=["time"])
    Path("output").mkdir(exist_ok=True)

    # Plot episode time
    episode_time_example(data)
    # Plot step time (overall & per seed)
    step_time_example(data)
    # Plot step time over intervals of 10 steps
    step_time_interval_example(data)
