from pathlib import Path

from dacbench.logger import load_logs, log2dataframe
from dacbench.plotting import plot_action
import matplotlib.pyplot as plt


def plot_scalar_action():
    file = Path("data/sigmoid_example/ActionFrequencyWrapper.jsonl")
    logs = load_logs(file)
    dataframe = log2dataframe(logs, wide=True)

    grid = plot_action(dataframe, interval=18, title="Sigmoid", col="seed", col_wrap=3)
    grid.savefig("output/sigmoid_example_action_interval_18.pdf")
    plt.show()

    grid = plot_action(dataframe, title="Sigmoid", col="seed", col_wrap=3)
    grid.savefig("output/sigmoid_example_action.pdf")
    plt.show()


def plot_action_modea():
    file = Path("data/ModeaBenchmark/ActionFrequencyWrapper.jsonl")
    logs = load_logs(file)
    dataframe = log2dataframe(logs, wide=True)

    grid = plot_action(dataframe, interval=5)
    grid.savefig("output/modea_action_interval_5.pdf")
    plt.show()

    grid = plot_action(dataframe)
    grid.savefig("output/modea_action.pdf")
    plt.show()


if __name__ == "__main__":
    plot_action_modea()
    plot_scalar_action()
