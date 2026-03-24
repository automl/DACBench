"""Get reference performance on a task."""

from __future__ import annotations

import itertools
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from carps.utils.index_configs import register_extra_paths
from hydra.core.hydra_config import HydraConfig

try:  # There have been breaking changes in CARP-S
    from carps.analysis.gather_data_utils import filelogs_to_df, normalize_logs
except ImportError:
    from carps.analysis.gather_data import filelogs_to_df, normalize_logs
import contextlib

from carps.analysis.utils import filter_only_final_performance
from carps.utils.env_vars import CARPS_ROOT
from carps.utils.running import optimize

with contextlib.suppress(ImportError):
    from carps.utils.index_configs import get_index_config
from hydra import compose, initialize_config_module

from dacbench.envs.dacboenv.utils.loggingutils import get_logger

logger = get_logger("ReferencePerformance")

if TYPE_CHECKING:
    from omegaconf import DictConfig


def is_slurm_cluster() -> bool:
    """Determine whether on a slurm cluster."""
    return any(shutil.which(cmd) for cmd in ["srun", "sbatch", "scontrol"])


class ReferencePerformance:
    """Reference Performance.

    Start and collect jobs if run data is missing.
    """

    def __init__(
        self,
        optimizer_id: str,
        task_ids: list[str],
        seeds: list[int],
        reference_performance_fn: str
        | Path = "reference_performance/reference_performance.parquet",
    ) -> None:
        """Init.

        Parameters
        ----------
        optimizer_id : str
            carps id of reference optimizer.
        task_ids : list[str]
            List of carps task ids.
        seeds : list[int]
            List of seeds.
        reference_performance_fn : str | Path, optional
            Filename of performance data, by default "reference_performance/reference_performance.parquet"
        """
        self.optimizer_id = optimizer_id
        self.task_ids = task_ids
        self.seeds = seeds
        self.reference_performance_fn = Path(reference_performance_fn)

        self.perf_df = lookup_performance(
            optimizer_id=self.optimizer_id,
            task_ids=self.task_ids,
            seeds=self.seeds,
            reference_performance_fn=self.reference_performance_fn,
        )

    def query_cost(self, optimizer_id: str, task_id: str, seed: int) -> float:
        """Query cost from reference performance data.

        Parameters
        ----------
        optimizer_id : str
            The optimizer id.
        task_id : str
            The task id.
        seed : int
            The seed.

        Returns:
        -------
        float
            Cost of final incumbent.
        """
        ids = [(optimizer_id, task_id, seed)]
        index_columns = ["optimizer_id", "task_id", "seed"]
        return (
            self.perf_df.set_index(index_columns)
            .loc[ids]
            .iloc[0]["trial_value__cost_inc"]
        )


def get_seed_override(seeds: list[int]) -> str:
    """Get seed override.

    Parameters
    ----------
    seeds : list[int]
        List of seeds.

    Returns:
    -------
    str
        Hydra override, e.g. `seed=1,2,3`.
    """
    return f"seed={','.join([str(s) for s in seeds])}"


def get_config_overrides(
    ids: list[str], index_csv_subpath: str, group_name: str, id_col: str
) -> list[str]:
    """Generic function to generate Hydra config overrides.

    Args:
        ids: list of task or optimizer IDs
        index_csv: path to CSV index file
        group_name: e.g. "task" or "optimizer"
        id_col: column name in CSV for the IDs, e.g. "task_id" or "optimizer_id"

    Returns:
    -------
        List of Hydra overrides like '+task/some/path=id1,id2'
    """
    try:  # If using hydra
        config = HydraConfig.get()
        index_paths = [
            Path(path_description["path"]) / index_csv_subpath
            for path_description in config["runtime"]["config_sources"]
            if path_description["schema"] == "file"
        ] + [CARPS_ROOT / "configs" / index_csv_subpath]
    except ValueError:
        index_paths = [CARPS_ROOT / "configs" / index_csv_subpath]
    if group_name == "task":
        register_extra_paths(
            [str(index_path.parent) for index_path in index_paths], None
        )
    elif group_name == "optimizer":
        register_extra_paths(
            None, [str(index_path.parent) for index_path in index_paths]
        )

    try:
        print(index_paths)
        df = pd.concat([get_index_config(path) for path in index_paths])  # noqa: PD901
    except NameError:
        df = pd.concat([pd.read_csv(path) for path in index_paths])  # noqa: PD901
    try:
        filtered = df.set_index(id_col).loc[ids].reset_index()
    except KeyError as e:
        raise KeyError(
            f"Probably one of {ids} is not native carps! Original error message: {e}"
        ) from e
    except Exception as e:
        raise e from e

    # Extract relative path and last element
    filtered["rel_path"] = filtered["config_fn"].map(
        lambda x: x.split(f"{group_name}/")[-1].replace(".yaml", "")
    )
    filtered["path"] = filtered["rel_path"].map(lambda x: "/".join(x.split("/")[:-1]))
    filtered["id_val"] = filtered["rel_path"].map(lambda x: x.split("/")[-1])

    return [
        f"+{group_name}/{path}={','.join(group['id_val'])}"
        for path, group in filtered.groupby("path")
    ]


def get_task_overrides(task_ids: list[str]) -> list[str]:
    """Get overrides for tasks.

    Parameters
    ----------
    task_ids : list[str]
        List of carps task ids.

    Returns:
    -------
    list[str]
        Hydra overrides.
    """
    return get_config_overrides(task_ids, "task/index.csv", "task", "task_id")


def get_optimizer_overrides(optimizer_ids: list[str]) -> list[str]:
    """Get optimizer overrides.

    Parameters
    ----------
    optimizer_ids : list[str]
        carps optimizer ids.

    Returns:
    -------
    list[str]
        Hydra overrides.
    """
    return get_config_overrides(
        optimizer_ids, "optimizer/index.csv", "optimizer", "optimizer_id"
    )


def group_tuples(tuples: list[tuple], depth: int = 0) -> list:
    """Recursively group tuples as much as possible.

    Flatten the last dimension to unique values.

    Parameters
    ----------
    tuples : list[tuple]
        List of tuples containing members.

    Returns:
    -------
    Group by appearance order. First group by first member of tuple, etc.
    """
    if not tuples:
        return []

    # Base case: last element
    if depth == len(tuples[0]) - 1:
        return sorted({t[depth] for t in tuples})

    # Group by the current depth
    grouped = defaultdict(list)
    for t in tuples:
        grouped[t[depth]].append(t)

    # Recurse for each group
    result = []
    for key, group in grouped.items():
        result.append([key, group_tuples(group, depth + 1)])
    return result


def lookup_performance(
    optimizer_id: str,
    task_ids: list[str],
    seeds: list[int],
    reference_performance_fn: str
    | Path = "reference_performance/reference_performance.parquet",
    n_processes: int | None = None,
) -> pd.DataFrame:
    """Lookup performance.

    If the performance file is not found, generate performance via jobs.

    If runs are missing, regenerate performance.

    Parameters
    ----------
    optimizer_id : str
        carps optimizer id.
    task_ids : list[str]
        carps task ids.
    seeds : list[int]
        Seeds.
    reference_performance_fn : str | Path, optional
        Filename of performance data, by default "reference_performance/reference_performance.parquet"
    n_processes : int | None, optional
        Number of processes for gathering data, by default None

    Returns:
    -------
    pd.DataFrame
        Reference performance dataframe, only containing final performance.
    """
    if not Path(reference_performance_fn).is_file():
        logger.info(f"No performance data found at {reference_performance_fn}")
        run_reference_optimizer(
            optimizer_id=optimizer_id,
            task_ids=task_ids,
            seeds=seeds,
            reference_performance_fn=reference_performance_fn,
            grouped_tuples=None,
            n_processes=n_processes,
        )

    logger.info(f"Loading performance data from {reference_performance_fn}")
    reference_df = pd.read_parquet(reference_performance_fn)
    combos = list(itertools.product([optimizer_id], task_ids, seeds))
    # Create a MultiIndex from combos (each element is a tuple)
    idx = pd.MultiIndex.from_tuples(combos, names=["optimizer_id", "task_id", "seed"])

    # DataFrame index
    df_index = reference_df.set_index(["optimizer_id", "task_id", "seed"]).index

    # Missing combos: those in `idx` but not in the DataFrame
    missing_idx = idx.difference(df_index)

    # Convert to list of tuples if needed
    missing_combos = list(missing_idx)
    if len(missing_combos) > 0:
        grouped_missing_combos = group_tuples(missing_combos)
        run_reference_optimizer(
            reference_performance_fn=reference_performance_fn,
            grouped_tuples=grouped_missing_combos,
            n_processes=n_processes,
        )
    reference_df = pd.read_parquet(reference_performance_fn)

    # Select matching rows
    return (
        reference_df.set_index(["optimizer_id", "task_id", "seed"])
        .loc[idx]
        .reset_index()
    )


def get_reference_performance(carps_overrides: list[str]) -> None:
    """Get reference performance of a carps optimizer.

    Run results go into the directory `reference_performance`.

    Parameters
    ----------
    carps_overrides : list[str]
        Carps overrides, e.g. ["+optimizer/smac20=blackbox", "+task/BBOB=cfg_2_1_0", "seed=1234"]
        Make sure to add one optimizer, a task, and the seed.
    """
    if not any(override.startswith("baserundir") for override in carps_overrides):
        carps_overrides.append("baserundir=reference_performance")

    with initialize_config_module(
        config_module="carps.configs", job_name="run_from_script"
    ):
        cfg: DictConfig = compose(config_name="base.yaml", overrides=carps_overrides)

    optimize(cfg=cfg)


def spawn_and_wait(commands: list[list[str]], poll_interval: float = 10.0) -> None:
    """Spawn multiple shell commands in the background and wait until all finish.

    Args:
        commands: List of commands, where each command is a list of strings (e.g. ["ls", "-la"]).
        poll_interval: Time in seconds between status checks.
    """
    # Spawn all processes
    for cmd in commands:
        logger.info(f"Spawning `{' '.join(cmd)}`...")
    processes = [subprocess.Popen(cmd) for cmd in commands]

    # Monitor processes until all finish
    while True:
        alive = [p.poll() is None for p in processes]  # True if still running
        if not any(alive):
            break
        logger.info(f"{sum(alive)} process(es) still running...")
        time.sleep(poll_interval)

    logger.info("All processes finished!")


def build_command(
    optimizer_override: str, task_override: str, seed: int | str, baserundir: str
) -> list[str]:
    """Build a single Hydra command for a given optimizer, task, and seed.

    Args:
        optimizer_override: Optimizer override, e.g. `+optimizer/smac20=blackbox`.
        task_override: Task override.
        seed: Seed value (int or string, e.g., 1 or "seed=1").
        baserundir: Base directory to store outputs.

    Returns:
    -------
        List of strings representing the full command to execute.
    """
    command = [
        "python",
        "-m",
        "carps.run",
        "hydra.searchpath=['pkg://dacbench/envs/dacboenv/configs','pkg://adaptaf/configs','pkg://optbench/configs']",
    ]
    overrides = [
        f"seed={seed}" if isinstance(seed, int) else seed,
        task_override,
        optimizer_override,
    ]
    if is_slurm_cluster():
        overrides.append("+cluster=cpu_noctua")
    overrides.append(f"baserundir={baserundir}")
    overrides.append("--multirun")
    command.extend(overrides)
    return command


def get_runcommands(
    optimizer_id: str | None = None,
    task_ids: list[str] | None = None,
    seeds: list[int] | None = None,
    baserundir: str = "reference_performance",
    grouped_tuples: list[list[list]] | None = None,
) -> list[list[str]]:
    """Generate Hydra commands to run reference optimizers on tasks with specified seeds.

    Supports two modes:
        1. Normal mode: Uses itertools.product to combine tasks, optimizer, and seed overrides.
        2. Grouped mode: Accepts a nested list of grouped tuples (from `group_tuples`) to minimize
           redundant combinations.

    Args:
        optimizer_id: Optional optimizer ID to run. If None, no optimizer override is applied.
        task_ids: Optional list of task IDs to run. If None, no task overrides are applied.
        seeds: Optional list of seeds. If None, no seed override is applied.
        reference_performance_fn: Path to reference performance parquet file (currently not used internally).
        baserundir: Base directory for storing outputs.
        n_processes: Number of parallel processes (currently not used internally).
        grouped_tuples: Optional nested list of grouped tuples from `group_tuples`.
                        Format: [[optimizer, [[task, [seeds]]]]].

    Returns:
    -------
        List of commands, where each command is a list of strings ready to execute via subprocess.
    """
    # Generate overrides for normal mode
    task_overrides = get_task_overrides(task_ids) if task_ids is not None else []
    optimizer_overrides = (
        get_optimizer_overrides([optimizer_id]) if optimizer_id is not None else []
    )
    seed_override = get_seed_override(seeds) if seeds is not None else ""

    runcommands = []

    if grouped_tuples is None:
        # Normal mode: use itertools.product
        for task_override, optimizer_override in itertools.product(
            task_overrides, optimizer_overrides
        ):
            runcommands.append(
                build_command(
                    optimizer_override, task_override, seed_override, baserundir
                )
            )
    else:
        # Grouped mode: iterate nested grouped tuples
        for optimizer_group in grouped_tuples:
            optimizer_name, task_groups = optimizer_group
            optimizer_override = get_optimizer_overrides([optimizer_name])[0]  # type: ignore[list-item] # pyright: ignore[reportArgumentType]
            for task_group in task_groups:
                task_name, seed_list = task_group
                task_override = get_task_overrides([task_name])[0]
                seed_override = get_seed_override(seed_list)
                runcommands.append(
                    build_command(
                        optimizer_override=optimizer_override,
                        task_override=task_override,
                        seed=seed_override,
                        baserundir=baserundir,
                    )
                )
    return runcommands


def run_reference_optimizer(
    optimizer_id: str | None = None,
    task_ids: list[str] | None = None,
    seeds: list[int] | None = None,
    reference_performance_fn: str | Path = "reference_performance.parquet",
    baserundir: str = "reference_performance",
    n_processes: int | None = 1,
    grouped_tuples: list[list[list]] | None = None,
) -> None:
    """Run reference optimizer on tasks and gather data.

    Args:
        optimizer_id: Optional optimizer ID to run. If None, no optimizer override is applied.
        task_ids: Optional list of task IDs to run. If None, no task overrides are applied.
        seeds: Optional list of seeds. If None, no seed override is applied.
        reference_performance_fn: Path to reference performance parquet file (currently not used internally).
        baserundir: Base directory for storing outputs.
        n_processes: Number of parallel processes (currently not used internally).
        grouped_tuples: Optional nested list of grouped tuples from `group_tuples`.
                        Format: [[optimizer, [[task, [seeds]]]]].
    """
    runcommands = get_runcommands(
        optimizer_id=optimizer_id,
        task_ids=task_ids,
        seeds=seeds,
        baserundir=baserundir,
        grouped_tuples=grouped_tuples,
    )
    spawn_and_wait(runcommands)

    logs, logs_cfg = filelogs_to_df(rundir=[baserundir], n_processes=n_processes)
    logs = normalize_logs(logs)

    ref_fn: Path = Path(reference_performance_fn)
    ref_fn.parent.mkdir(parents=True, exist_ok=True)
    reference_df = filter_only_final_performance(logs)
    reference_df.to_parquet(ref_fn)
    logs_cfg.to_parquet(ref_fn.with_stem(ref_fn.stem + "_cfg"))
