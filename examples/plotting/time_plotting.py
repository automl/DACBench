from pathlib import Path
import pandas as pd
from dacbench.logger import load_logs, log2dataframe
from dacbench.plotting import plot_step_time, plot_episode_time
import matplotlib.pyplot as plt


def step_time_example(data):
    grid = plot_step_time(data, y_label="Step Duration [s]")
    grid.savefig("output/step_duration.pdf")
    plt.show()

    grid = plot_step_time(data, y_label="Step Duration [s]", hue="seed")
    grid.savefig("output/step_duration_per_seed.pdf")
    plt.show()


def episode_time_example(data):
    print(data[~data.episode_duration.isna()])
    grid = plot_episode_time(
        data[~data.episode_duration.isna()], y_label="Episode Duration [s]"
    )
    grid.savefig("output/episode_duration.pdf")
    plt.show()


def step_time_interval_example(data: pd.DataFrame, interval: int = 10):
    """

    Parameters
    ----------
    data : pd.DataFrame
        The non-wide data frame resulting from loading the logging results from EpisodeTimeTracker
    interval : int
        Number of steps to average over

    """
    grid = plot_step_time(data, interval, title="Mean Step Duration")
    grid.savefig("output/step_duration.pdf")
    plt.show()


if __name__ == "__main__":
    file = Path("data/sigmoid_example/EpisodeTimeWrapper.jsonl")
    logs = load_logs(file)
    data = log2dataframe(logs, wide=True, drop_columns=["time"])
    episode_time_example(data)
    step_time_example(data)
    step_time_interval_example(data)
