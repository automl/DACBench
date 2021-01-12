import numpy as np
import seaborn as sns
import pandas as pd

sns.set_style("darkgrid")


def space_sep_upper(column_name):
    if column_name is None:
        return None
    return column_name.title().replace("_", " ")


def generate_global_step(
    data, x_column="global_step", x_label_columns=["episode", "step"]
):
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


def add_multi_level_ticks(grid, plot_index, x_column, x_label_columns):
    for ax in grid.axes.flat:
        ticks = ax.get_xticks()
        sub_set = plot_index[plot_index[x_column].isin(ticks)]
        new_labels = (
            sub_set.loc[tick][x_label_columns].tolist()
            if tick in sub_set.index
            else (None, None)
            for tick in ticks
        )
        new_labels = [
            f"{epoch}:{step}" if epoch is not None else "" for epoch, step in new_labels
        ]
        ax.set_xticklabels(new_labels, minor=False)


def plot(
    plot_function, settings, title=None, x_label=None, y_label=None, **args
) -> sns.FacetGrid:
    settings.update(args.items())
    grid = plot_function(**settings)

    x_label = space_sep_upper(grid._x_var) if x_label is None else x_label
    y_label = space_sep_upper(grid._y_var) if y_label is None else y_label
    grid.set_xlabels(x_label)
    grid.set_ylabels(y_label)
    grid.tight_layout()
    if title is not None:
        grid.fig.suptitle(title, y=0.97)
        grid.fig.subplots_adjust(top=0.9)

    return grid


def plot_performance(
    data, title=None, x_label=None, y_label=None, **args
) -> sns.FacetGrid:
    settings = {
        "data": data,
        "x": "episode",
        "y": "overall_performance",
        "kind": "line",
    }

    grid = plot(sns.relplot, settings, title, x_label, y_label, **args)

    return grid


def plot_performance_per_instance(
    data, title=None, x_label=None, y_label=None, **args
) -> sns.FacetGrid:
    order = data.groupby("instance").mean().sort_values("overall_performance").index
    settings = {
        "data": data,
        "x": "instance",
        "y": "overall_performance",
        "order": order,
        "kind": "bar",
    }
    grid = plot(sns.catplot, settings, title, x_label, y_label, **args)
    grid.set_titles("Mean Performance per Instance")
    return grid


def plot_step_time(
    data, interval=1, title=None, x_label="Epoch:Step", y_label=None, **args
) -> sns.FacetGrid:
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

    grid = plot(sns.relplot, settings, title, x_label, y_label, **args)
    add_multi_level_ticks(grid, plot_index, x_column, x_label_columns)

    return grid


def plot_episode_time(
    data, title=None, x_label=None, y_label=None, **args
) -> sns.FacetGrid:
    settings = {
        "data": data,
        "x": "episode",
        "y": "episode_duration",
        "kind": "line",
    }

    grid = plot(sns.relplot, settings, title, x_label, y_label, **args)

    return grid


def plot_action(
    data,
    show_global_step=False,
    interval=1,
    title=None,
    x_label=None,
    y_label=None,
    **args,
):
    return plot_space(
        data, "action", show_global_step, interval, title, x_label, y_label, **args
    )


def plot_state(
    data,
    show_global_step=False,
    interval=1,
    title=None,
    x_label=None,
    y_label=None,
    **args,
):
    return plot_space(
        data, "state", show_global_step, interval, title, x_label, y_label, **args
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

    space_entries = list(
        filter(lambda col: col.startswith(space_column_name), data.columns)
    )
    number_of_space_entries = len(space_entries)
    y_label_name = space_column_name
    if number_of_space_entries > 1:
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

        space_column_name, *_ = space_entries

    data, plot_index, x_column, x_label_columns = generate_global_step(data)

    if interval > 1:
        data["interval"] = data[x_column] // interval
        group_columns = list(
            data.columns.drop(x_label_columns + [x_column, space_column_name])
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
    if number_of_space_entries > 1:
        settings["col"] = "i"
    if number_of_space_entries > 3:
        settings["col_wrap"] = 3

    if "seed" in data.columns:
        settings["hue"] = "instance"

    if x_label is None:
        x_label = None if show_global_step else "Epoch:Step"

    if y_label is None:
        y_label = y_label_name
    grid = plot(sns.relplot, settings, title, x_label, y_label, **args)
    if not show_global_step:
        add_multi_level_ticks(grid, plot_index, x_column, x_label_columns)

    return grid
