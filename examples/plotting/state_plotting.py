"""Example for plotting of states."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from dacbench.logger import load_logs, log2dataframe
from dacbench.plotting import plot_state


def plot_state_CMAES():  # noqa: N802
    """Plot state information of CMA-ES run over time"""
    # Since converting the json logs to a data frame takes a couple of minutes
    # we we cache the logs for tuning the plot settings in a picked datafarme object
    path = Path("output/cached_logs.pickle")

    if not path.exists():
        file = Path("./data/CMAESBenchmark/StateTrackingWrapper.jsonl")
        if not file.exists():
            print(
                "Please run 'examples/benchmarks/chainerrl_cma.py' to generate plotting data first"
            )
            return
        logs = load_logs(file)
        dataframe = log2dataframe(logs, wide=True)
        dataframe.to_pickle(path)
    else:
        dataframe = pd.read_pickle(path)
    Path("output").mkdir(exist_ok=True)

    # The CMAES observation space has over 170 dims. Here we just plot a subset
    # here we get all different parts of the states
    columns = pd.DataFrame(
        (column.split("_") for column in dataframe.columns),
        columns=["part", "subpart", "i"],
    )
    state_parts = columns[columns["part"] == "state"]["subpart"].unique()
    print(f"State parts {state_parts}")

    # But since History Deltas(80), Past Deltas(40) and Past Sigma Deltas(40)
    # have to many dims to be plotted we only show
    state_parts = ["Loc", "Population Size", "Sigma"]

    for state_part in state_parts:
        state_part_columns = [
            column
            for column in dataframe.columns
            if not column.startswith("state") or column.split("_")[1] == state_part
        ]
        grid = plot_state(dataframe[state_part_columns], interval=100, title=state_part)
        grid.savefig(f"output/cmaes_state_{state_part}.pdf")
        plt.show()

    # one can also show the global step (increasing step over episodes) on x axis
    grid = plot_state(
        dataframe[state_part_columns],
        show_global_step=True,
        interval=100,
        title=state_part,
    )
    grid.savefig(f"output/cmaes_state_{state_part}_global_step.pdf")
    plt.show()


if __name__ == "__main__":
    plot_state_CMAES()
