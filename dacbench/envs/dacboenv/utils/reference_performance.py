"""Get reference performance on a task (carps-free)."""

from __future__ import annotations

import itertools
from collections import defaultdict
from pathlib import Path

import pandas as pd

from dacbench.envs.dacboenv.utils.carps_optimizer import build_smac_facade
from dacbench.envs.dacboenv.utils.loggingutils import get_logger

logger = get_logger("ReferencePerformance")


class ReferencePerformance:
    """Reference Performance.

    Load and query precomputed performance data.
    """

    def __init__(
        self,
        optimizer_id: str,
        task_ids: list[str],
        seeds: list[int] | None,
        reference_performance_fn: str
        | Path = "reference_performance/reference_performance.parquet",
        n_trials: int = 77,
    ) -> None:
        """Init.

        Parameters
        ----------
        optimizer_id : str
            Optimizer id of reference optimizer.
        task_ids : list[str]
            List of task ids.
        seeds : list[int]
            List of seeds.
        reference_performance_fn : str | Path, optional
            Filename of performance data, by default "reference_performance/reference_performance.parquet"
        n_trials : int, optional
            Maximum number of optimization trials, by default 77.
        """
        self.optimizer_id = optimizer_id
        self.task_ids = task_ids
        self.seeds = seeds if seeds is not None else list(range(1, 21))
        self.reference_performance_fn = Path(reference_performance_fn)
        self.n_trials = n_trials

        self.perf_df = lookup_performance(
            optimizer_id=self.optimizer_id,
            task_ids=self.task_ids,
            seeds=self.seeds,
            reference_performance_fn=self.reference_performance_fn,
            n_trials=self.n_trials,
        )

    def query_cost(
        self,
        optimizer_id: str,
        task_id: str,
        seed: int | None,
        key_performance: str = "trial_value__cost_inc",
    ) -> float:
        """Query cost from reference performance data.

        Parameters
        ----------
        optimizer_id : str
            The optimizer id.
        task_id : str
            The task id.
        seed : int | None
            The seed. If None, averages over all precomputed seeds.
        key_performance : str, optional
            Column name for the performance metric, by default "trial_value__cost_inc".

        Returns:
        -------
        float
            Cost of final incumbent (or mean over seeds when seed is None).
        """
        if seed is None:
            ids = [(optimizer_id, task_id)]
            index_columns = ["optimizer_id", "task_id"]
            return (
                self.perf_df.set_index(index_columns).loc[ids][key_performance].mean()
            )
        ids = [(optimizer_id, task_id, seed)]
        index_columns = ["optimizer_id", "task_id", "seed"]
        return self.perf_df.set_index(index_columns).loc[ids].iloc[0][key_performance]


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
    n_trials: int = 77,
) -> pd.DataFrame:
    """Lookup performance.

    If the performance file is not found, generate performance via SMAC3 runs.

    If runs are missing, regenerate performance.

    Parameters
    ----------
    optimizer_id : str
        Optimizer id.
    task_ids : list[str]
        Task ids.
    seeds : list[int]
        Seeds.
    reference_performance_fn : str | Path, optional
        Filename of performance data, by default "reference_performance/reference_performance.parquet"
    n_processes : int | None, optional
        Number of processes for gathering data, by default None
    n_trials : int, optional
        Maximum number of optimization trials, by default 77.

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
            n_trials=n_trials,
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
            n_trials=n_trials,
        )
    reference_df = pd.read_parquet(reference_performance_fn)

    # Select matching rows
    return (
        reference_df.set_index(["optimizer_id", "task_id", "seed"])
        .loc[idx]
        .reset_index()
    )


def run_reference_optimizer(
    optimizer_id: str | None = None,
    task_ids: list[str] | None = None,
    seeds: list[int] | None = None,
    reference_performance_fn: str | Path = "reference_performance.parquet",
    n_trials: int = 77,
    baserundir: str = "reference_performance",
    n_processes: int | None = 1,
    grouped_tuples: list[list[list]] | None = None,
) -> None:
    """Run reference SMAC3 optimizer on tasks and gather data.

    Args:
        optimizer_id: Optional optimizer ID to run. If None, no optimizer override is applied.
        task_ids: Optional list of task IDs to run. If None, no task overrides are applied.
        seeds: Optional list of seeds. If None, no seed override is applied.
        reference_performance_fn: Path to reference performance parquet file.
        n_trials: Maximum number of optimization trials per run.
        baserundir: Base directory for storing outputs (unused, kept for API compat).
        n_processes: Number of parallel processes (unused, kept for API compat).
        grouped_tuples: Optional nested list of grouped tuples from `group_tuples`.
                        Format: [[optimizer, [[task, [seeds]]]]].
    """
    # Parse grouped tuples if provided (for efficiency with shared seeds/tasks)
    if grouped_tuples is not None:
        # Flatten grouped tuples back to (optimizer_id, task_id, seed) triplets
        triplets = []
        for optimizer_group in grouped_tuples:
            optimizer_name = optimizer_group[0]
            for task_group in optimizer_group[1]:
                task_name = task_group[0]
                for seed in task_group[1]:
                    triplets.append((optimizer_name, task_name, seed))
    else:
        triplets = []
        for task_id in task_ids or []:
            for seed in seeds or []:
                triplets.append((optimizer_id or "SMAC3-BlackBoxFacade", task_id, seed))

    # Run SMAC3 for each triplet
    records = []
    for opt_id, task_id, seed in triplets:
        facade = build_smac_facade(
            task_id=task_id,
            seed=seed,
            n_trials=n_trials,
            optimizer_cfg=None,  # Use default config
        )
        facade.optimize()
        trajectory = facade.intensifier.trajectory
        final_cost = trajectory[-1].costs[0] if trajectory else float("inf")
        records.append(
            {
                "optimizer_id": opt_id,
                "task_id": task_id,
                "seed": seed,
                "trial_value__cost_inc": final_cost,
            }
        )

    ref_fn = Path(reference_performance_fn)
    ref_fn.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_parquet(ref_fn) if ref_fn.is_file() else pd.DataFrame()
    pd.concat([existing, pd.DataFrame(records)], ignore_index=True).to_parquet(ref_fn)
