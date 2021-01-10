from pathlib import Path

from dacbench.logger import load_logs, log2dataframe
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")


def space_sep_upper(column_name):
    return column_name.title().replace("_", " ")


def plot(plot_function, settings, x_label=None, y_label=None, **args):
    settings.update(args.items())
    grid = plot_function(**settings)

    x_label = space_sep_upper(grid._x_var) if x_label is None else x_label
    y_label = space_sep_upper(grid._y_var) if y_label is None else y_label
    grid.set_xlabels(x_label)
    grid.set_ylabels(y_label)
    grid.tight_layout()

    return grid


def plot_performance(data, x_label=None, y_label=None, **args):

    settings = {
        "data": data,
        "x": "episode",
        "y": "overall_performance",
        "kind": "line",
    }

    grid = plot(sns.relplot, settings, x_label, y_label, **args)

    return grid


def plot_performance_per_instance(data, x_label=None, y_label=None, **args):
    settings = {
        "data": data,
        "x": "instance",
        "y": "overall_performance",
        "kind": "bar",
    }
    grid = plot(sns.catplot, settings, x_label, y_label, **args)
    grid.set_titles("Mean Performance per Instance")
    return grid


if __name__ == "__main__":

    path = Path("output/LubyBenchmark/optimal_1/PerformanceTrackingWrapper.jsonl")
    logs = load_logs(path)
    data = log2dataframe(logs, wide=True, drop_columns=["time"])
    grid = plot_performance_per_instance(data)

    grid.savefig("fig.pdf")
    plt.show()
