"""Plotting helper."""
from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("darkgrid")


def space_sep_upper(column_name: str) -> str:
    """Separates strings at underscores into headings.
    Used to generate labels from logging names.

    Parameters
    ----------
    column_name : str
        Name to generate label for

    Returns:
    -------
    str

    """
    if column_name is None:
        return None
    return column_name.title().replace("_", " ")


def generate_global_step(
    data: pd.DataFrame,
    x_column: str = "global_step",
    x_label_columns: str = ["episode", "step"],
) -> tuple[pd.DataFrame, str, list[str]]:
    """Add a global_step column which enumerate all step over all episodes.

    Returns the altered data, a data frame containing mapping between global_step,
    x_column and x_label_columns.

    Often used in combination with add_multi_level_ticks.

    Parameters
    ----------
    data: pd.DataFrame
        data source
    x_column: str
        the name of the global_step (default 'global_step')
    x_label_columns: [str, ...]
        the name and hierarchical order of the columns (default ['episode', 'step']

    Returns:
    -------
    (data, plot_index, x_column, x_label_columns)

    """
    plot_index = (
        data.groupby(x_label_columns)
        .count()
        .reset_index()[x_label_columns]
        .sort_values(x_label_columns)
    )
    plot_index[x_column] = np.arange(len(plot_index))
    plot_index.set_index(x_column)
    data = data.merge(plot_index, on=x_label_columns)
    return data, plot_index, x_column, x_label_columns


def add_multi_level_ticks(
    grid: sns.FacetGrid, plot_index: pd.DataFrame, x_column: str, x_label_columns: str
) -> None:
    """Expects a FacedGrid with global_step (x_column) as x-axis
    and replaces the tick labels to match format episode:step.

    E.g. Run with 3 episodes, each of 10 steps. This results in 30 global steps.
    The resulting tick labels could be ['0', '4', '9', '14', '19', '24', '29'].
    After applying this method they will look like
    ['0:0', '0:4', '1:0', '1:4', '2:0', '2:4', '3:0', '3:4']

    Parameters
    ----------
    grid: sns.FacesGrid
        The grid to plot onto
    plot_index: pd.DataFrame
        The mapping between current tick labels (global step values) and new tick labels
        joined by ':'. Usually the result from generate_global_step
    x_column: str
        column label to use for looking up tick values
    x_label_columns: [str, ...]
        columns labels of columns to use for new labels (joined by ':'

    """
    for ax in grid.axes.flat:
        ticks = ax.get_xticks()
        sub_set = plot_index[plot_index[x_column].isin(ticks)]
        new_labels = (
            (
                sub_set.loc[tick][x_label_columns].tolist()
                if tick in sub_set.index
                else (None, None)
            )
            for tick in ticks
        )
        new_labels = [
            f"{epoch}:{step}" if epoch is not None else "" for epoch, step in new_labels
        ]
        ax.set_xticklabels(new_labels, minor=False)


def plot(
    plot_function,
    settings: dict,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    **kwargs,
) -> sns.FacetGrid:
    """Helper function that creates a FacetGrid.

    1. Updates settings with kwargs (overwrites values)
    2. Plots using plot_function(**settings)
    3. Set x and y labels of not provided the columns names will converted
      to pretty strings using space_sep_upper
    4. Sets title (some times has to be readjusted afterwards especially in
      case of large plots e.g. multiple rows/cols)

    Parameters
    ----------
    plot_function:
        function to generate the FacedGrid. E.g. sns.catplot or sns.catplot
    settings: dict
        a dicts containing all needed default settings.
    title: str
        Title of the plot (optional)
    x_label: str
        Label of the x-axis (optional)
    y_label: str
        Label of the y-axis (optional)
    kwargs:
        Keyword arguments to overwrite default settings.

    Returns:
    -------
    sns.FacedGrid

    """
    settings.update(kwargs.items())  # 1.
    grid = plot_function(**settings)  # 2.

    # 3.
    x_label = space_sep_upper(grid._x_var) if x_label is None else x_label
    y_label = space_sep_upper(grid._y_var) if y_label is None else y_label
    grid.set_xlabels(x_label)
    grid.set_ylabels(y_label)

    # 4.
    grid.tight_layout()
    if title is not None:
        grid.fig.suptitle(title, y=0.97)  # rule of thumb. Has to be improved in future
        grid.fig.subplots_adjust(top=0.9)

    return grid


def plot_performance(
    data, title=None, x_label=None, y_label=None, **kwargs
) -> sns.FacetGrid:
    """Create a line plot of the performance over episodes.

    Per default the mean performance and and one stddev over all
    instances and seeds is shown if you want to change this specify a property
    to map those attributes to e.g hue='seed' or/and col='instance'.
    For more details see: https://seaborn.pydata.org/generated/seaborn.relplot.html

    For examples refer to examples/plotting/performance_plotting.py

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe resulting from logging and loading using
        log2dataframe(logs, wide=True)
    title: str
        Title of the plot (optional)
    x_label: str
        Label of the x-axis (optional)
    y_label: str
        Label of the y-axis (optional)
    kwargs:
        Keyword arguments to overwrite default settings.

    Returns:
    -------
    sns.FacedGrid

    """
    settings = {
        "data": data,
        "x": "episode",
        "y": "overall_performance",
        "kind": "line",
    }
    return plot(sns.relplot, settings, title, x_label, y_label, **kwargs)


def plot_performance_per_instance(
    data, title=None, x_label=None, y_label=None, **args
) -> sns.FacetGrid:
    """Create bar plot of the mean performance per instance ordered by the performance.

    Per default the mean performance seeds is shown if you want to change
    this specify a property to map seed to e.g. col='seed'.
    For more details see: https://seaborn.pydata.org/generated/seaborn.catplot.html

    For examples refer to examples/plotting/performance_plotting.py

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe resulting from logging and loading using
        log2dataframe(logs, wide=True)
    title: str
        Title of the plot (optional)
    x_label: str
        Label of the x-axis (optional)
    y_label: str
        Label of the y-axis (optional)
    kwargs:
        Keyword arguments to overwrite default settings.

    Returns:
    -------
    sns.FacedGrid

    """
    # order the columns by mean instance
    order = data.groupby("instance").mean().sort_values("overall_performance").index
    settings = {
        "data": data,
        "x": "instance",
        "y": "overall_performance",
        "order": order,
        "kind": "bar",
    }
    grid = plot(sns.catplot, settings, title, x_label, y_label, **args)
    # todo: should probably not always be set like this (multi row/col)
    grid.set_titles("Mean Performance per Instance")
    return grid


def plot_step_time(
    data,
    show_global_step=False,
    interval=1,
    title=None,
    x_label=None,
    y_label=None,
    **args,
) -> sns.FacetGrid:
    """Create a line plot showing the measured time per step.

    Per default the mean performance and and one stddev over all instances
    and seeds is shown if you want to change this specify a property to map
    those attributes to e.g hue='seed' or/and col='instance'.
    For more details see: https://seaborn.pydata.org/generated/seaborn.relplot.html

    For examples refer to examples/plotting/time_plotting.py

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe resulting from logging and loading using
        log2dataframe(logs, wide=True)
    show_global_step: bool
        If to show the global_step (step enumerated over all episodes)
        or Episode:Step. (False default)
    interval: int
        Interval in number of steps to average over. (default = 1)
    title: str
        Title of the plot (optional)
    x_label: str
        Label of the x-axis (optional)
    y_label: str
        Label of the y-axis (optional)
    kwargs:
        Keyword arguments to overwrite default settings.

    Returns:
    -------
    sns.FacedGrid

    """
    multi_level_x_label = "Epoch:Step"
    data, plot_index, x_column, x_label_columns = generate_global_step(data)
    if interval > 1:
        data["groups"] = data[x_column] // interval
        data = data.groupby("groups").agg({x_column: "min", "step_duration": "mean"})
        y_label = (
            f"Mean per duration per {interval} steps" if y_label is None else y_label
        )

    settings = {
        "data": data,
        "x": x_column,
        "y": "step_duration",
        "kind": "line",
    }
    if x_label is None and not show_global_step:
        x_label = multi_level_x_label

    grid = plot(sns.relplot, settings, title, x_label, y_label, **args)
    if not show_global_step:
        add_multi_level_ticks(grid, plot_index, x_column, x_label_columns)

    return grid


def plot_episode_time(
    data, title=None, x_label=None, y_label=None, **kargs
) -> sns.FacetGrid:
    """Create a line plot showing the measured time per episode.

    Per default the mean performance and and one stddev over all instances
    and seeds is shown if you want to change this specify a property to map
    those attributes to e.g hue='seed' or/and col='instance'.
    For more details see: https://seaborn.pydata.org/generated/seaborn.relplot.html

    For examples refer to examples/plotting/time_plotting.py

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe resulting from logging and loading using
        log2dataframe(logs, wide=True)
    title: str
        Title of the plot (optional)
    x_label: str
        Label of the x-axis (optional)
    y_label: str
        Label of the y-axis (optional)
    kwargs:
        Keyword arguments to overwrite default settings.

    Returns:
    -------
    sns.FacedGrid

    """
    settings = {
        "data": data,
        "x": "episode",
        "y": "episode_duration",
        "kind": "line",
    }

    return plot(sns.relplot, settings, title, x_label, y_label, **kargs)


def plot_action(
    data,
    show_global_step=False,
    interval=1,
    title=None,
    x_label=None,
    y_label=None,
    **kargs,
):
    """Create a line plot showing actions over time.

    Please be aware that spaces can be quite large and the plots can become quite messy
    (and take some time) if you try plot all dimensions at once.
    It is therefore recommended to select a subset of columns before running the
    plot method. Especially for dict spaces.

    Per default the mean performance and and one stddev over all instances
    and seeds is shown if you want to change this specify a property to map those
    attributes to e.g hue='seed' or/and col='instance'.
    For more details see: https://seaborn.pydata.org/generated/seaborn.relplot.html

    For examples refer to examples/plotting/action_plotting.py

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe resulting from logging and loading using
        log2dataframe(logs, wide=True)
    show_global_step: bool
        If to show the global_step (step enumerated over all episodes)
        or Episode:Step. (False default)
    interval: int
        Interval in number of steps to average over. (default = 1)
    title: str
        Title of the plot (optional)
    x_label: str
        Label of the x-axis (optional)
    y_label: str
        Label of the y-axis (optional)
    kwargs:
        Keyword arguments to overwrite default settings.

    Returns:
    -------
    sns.FacedGrid

    """
    return plot_space(
        data, "action", show_global_step, interval, title, x_label, y_label, **kargs
    )


def plot_state(
    data,
    show_global_step=False,
    interval=1,
    title=None,
    x_label=None,
    y_label=None,
    **kargs,
):
    """Create a line plot showing state over time.

    -----
    Create a line plot showing space over time.

    Please be aware that spaces can be quite large and the plots can become quite messy
    (and take some time) if you try plot all dimensions at once.
    It is therefore recommended to select a subset of columns before running the
    plot method. Especially for dict spaces.

    Per default the mean performance and and one stddev over all instances
    and seeds is shown if you want to change this specify a property to map those
    attributes to e.g hue='seed' or/and col='instance'.
    For more details see: https://seaborn.pydata.org/generated/seaborn.relplot.html

    For examples refer to examples/plotting/state_plotting.py

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe resulting from logging and loading using
        log2dataframe(logs, wide=True)
    show_global_step: bool
        If to show the global_step (step enumerated over all episodes)
        or Episode:Step. (False default)
    interval: int
        Interval in number of steps to average over. (default = 1)
    title: str
        Title of the plot (optional)
    x_label: str
        Label of the x-axis (optional)
    y_label: str
        Label of the y-axis (optional)
    kwargs:
        Keyword arguments to overwrite default settings.

    Returns:
    -------
    sns.FacedGrid

    """
    return plot_space(
        data, "state", show_global_step, interval, title, x_label, y_label, **kargs
    )


def plot_space(
    data,
    space_column_name,
    show_global_step,
    interval=1,
    title=None,
    x_label=None,
    y_label=None,
    **args,
) -> sns.FacetGrid:
    """Create a line plot showing space over time.

    Please be aware that spaces can be quite large and the plots can become quite messy
    (and take some time) if you try plot all dimensions at once.
    It is therefore recommended to select a subset of columns before running the
    plot method. Especially for dict spaces.

    Per default the mean performance and and one stddev over all instances
    and seeds is shown if you want to change this specify a property to map those
    attributes to e.g hue='seed' or/and col='instance'.
    For more details see: https://seaborn.pydata.org/generated/seaborn.relplot.html

    For examples refer to
       examples/plotting/state_plotting.py or
       examples/plotting/action_plotting.py


    Parameters
    ----------
    data: pd.DataFrame
        Dataframe resulting from logging and loading
        using log2dataframe(logs, wide=True)
    space_column_name : str
        Name of the column in the space which to plot
    show_global_step: bool
        If to show the global_step (step enumerated over all episodes)
        or Episode:Step. (False default)
    interval: int
        Interval in number of steps to average over. (default = 1)
    title: str
        Title of the plot (optional)
    x_label: str
        Label of the x-axis (optional)
    y_label: str
        Label of the y-axis (optional)
    kwargs:
        Keyword arguments to overwrite default settings.

    Returns:
    -------
    sns.FacedGrid

    """
    # first find columns with prefix space_column_name
    space_entries = list(
        filter(lambda col: col.startswith(space_column_name), data.columns)
    )
    number_of_space_entries = len(space_entries)
    y_label_name = space_column_name

    if number_of_space_entries > 1:
        # if we have more than one space dims we reshape the dataframe
        # in order to be able to control the plots behavior per dimension
        data = pd.wide_to_long(
            data,
            stubnames=[space_column_name],
            sep="_",
            i=["episode", "step", "instance"]
            + (["seed"] if "seed" in data.columns else []),
            j="i",
            suffix=".*",
        ).reset_index()
    elif number_of_space_entries == 1 and space_column_name not in data.columns:
        # Of there is only one dimension but the name is odd
        space_column_name, *_ = space_entries

    data, plot_index, x_column, x_label_columns = generate_global_step(data)

    # perform averaging over intervals
    if interval > 1:
        data["interval"] = data[x_column] // interval
        group_columns = list(
            data.columns.drop([*x_label_columns, x_column, space_column_name])
        )
        data = data.groupby(group_columns).agg(
            {x_column: "min", space_column_name: "mean"}
        )
        y_label = (
            f"Mean {y_label_name} per {interval} steps" if y_label is None else y_label
        )
        data = data.reset_index()

    settings = {
        "data": data,
        "x": x_column,
        "y": space_column_name,
        "kind": "line",
    }

    # we want the different dims in different plots / columns
    # todo: refactor
    if number_of_space_entries > 1:
        settings["col"] = "i"
    if number_of_space_entries > 3:
        settings["col_wrap"] = 3

    if "instance" in data.columns:
        settings["hue"] = "instance"

    if x_label is None:
        x_label = None if show_global_step else "Epoch:Step"

    if y_label is None:
        y_label = y_label_name

    grid = plot(sns.relplot, settings, title, x_label, y_label, **args)
    if not show_global_step:
        add_multi_level_ticks(grid, plot_index, x_column, x_label_columns)

    return grid
