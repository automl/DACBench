"""Build SMAC3 optimizer facade (carps-free)."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

import ioh
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter
from omegaconf import DictConfig, OmegaConf
from smac.scenario import Scenario

if TYPE_CHECKING:
    from smac.facade.abstract_facade import AbstractFacade


def _parse_bbob_task_id(task_id: str) -> tuple[int, int, int]:
    """Parse 'bbob/{dim}/{func_id}/{instance}' -> (dim, func_id, instance)."""
    parts = task_id.split("/")
    assert parts[0] == "bbob", f"Only 'bbob' benchmark supported, got '{parts[0]}'"
    return int(parts[1]), int(parts[2]), int(parts[3])


def _build_bbob_configspace(dim: int) -> ConfigurationSpace:
    cs = ConfigurationSpace()
    for i in range(dim):
        cs.add_hyperparameter(UniformFloatHyperparameter(f"x{i}", lower=-5.0, upper=5.0))
    return cs


def _build_bbob_target_fn(func_id: int, instance: int, dim: int):
    """Create target function from ioh problem.

    ioh is 1-based for instances; carps task IDs are 0-based.
    """
    problem = ioh.get_problem(fid=func_id, instance=instance + 1, dimension=dim)

    def target_fn(config: Configuration, seed: int = 0) -> float:
        return problem([config[f"x{i}"] for i in range(dim)])

    return target_fn


def _instantiate_from_target(target: str):
    """Instantiate a class from a dotted _target_ string."""
    mod_path, _, cls_name = target.rpartition(".")
    mod = import_module(mod_path)
    return getattr(mod, cls_name)


def build_smac_facade(
    task_id: str,
    seed: int,
    n_trials: int,
    optimizer_cfg: DictConfig | None = None,
) -> AbstractFacade:
    """Build a SMAC3 AbstractFacade for the given task.

    Parameters
    ----------
    task_id : str
        The task id (e.g. 'bbob/14/0/5').
    seed : int
        The seed.
    n_trials : int
        Maximum number of optimization trials.
    optimizer_cfg : DictConfig, optional
        Optimizer config (must have smac_cfg with scenario, smac_class, etc.).

    Returns:
    -------
    AbstractFacade
        The SMAC3 facade ready for optimization.
    """
    # Ensure task_id is a native string (avoid numpy.str_ issues with OmegaConf)
    task_id = str(task_id)

    # Resolve interpolations against parent config
    parent = OmegaConf.create({
        "benchmark_id": task_id.split("/")[1],
        "task_id": task_id,
        "seed": seed,
        "outdir": "runs/test",
    })
    if optimizer_cfg is not None:
        optimizer_cfg = OmegaConf.merge(parent, optimizer_cfg)
    else:
        optimizer_cfg = parent

    dim, func_id, instance = _parse_bbob_task_id(task_id)
    cs = _build_bbob_configspace(dim)
    target_fn = _build_bbob_target_fn(func_id, instance, dim)

    # Parse scenario config (explicit n_trials overrides any in smac_cfg)
    scenario_raw = OmegaConf.select(optimizer_cfg, "smac_cfg.scenario")
    if scenario_raw is None:
        scenario_raw = OmegaConf.create()
    scenario_cfg = OmegaConf.to_container(scenario_raw, resolve=True)
    scenario_cfg["n_trials"] = n_trials
    scenario = Scenario(cs, seed=seed, **scenario_cfg)

    # Parse smac_class
    smac_class_path = OmegaConf.select(
        optimizer_cfg, "smac_cfg.smac_class"
    ) or "smac.facade.blackbox_facade.BlackBoxFacade"
    facade_class = _instantiate_from_target(smac_class_path)

    # Parse initial_design
    id_cfg = OmegaConf.select(
        optimizer_cfg, "smac_cfg.smac_kwargs.initial_design"
    )
    initial_design = None
    if id_cfg is not None:
        id_cfg_dict = OmegaConf.to_container(id_cfg, resolve=True)
        id_cls = _instantiate_from_target(id_cfg_dict.pop("_target_"))
        if "max_ratio" not in id_cfg_dict:
            id_cfg_dict["max_ratio"] = 0.2
        if "n_configs" not in id_cfg_dict or id_cfg_dict["n_configs"] is None:
            id_cfg_dict["n_configs"] = None
        initial_design = id_cls(scenario, **id_cfg_dict)

    # Parse random_design
    rd_cfg = OmegaConf.select(
        optimizer_cfg, "smac_cfg.smac_kwargs.random_design"
    )
    random_design = None
    if rd_cfg is not None:
        rd_cfg_dict = OmegaConf.to_container(rd_cfg, resolve=True)
        rd_cls = _instantiate_from_target(rd_cfg_dict.pop("_target_"))
        # Convert scenario ref back to object
        rd_cfg_dict["scenario"] = scenario
        random_design = rd_cls(scenario, **rd_cfg_dict)

    # Parse acquisition_function
    acq_cfg = OmegaConf.select(optimizer_cfg, "smac_cfg.smac_kwargs.acquisition_function")
    acq_fn = None
    if acq_cfg is not None:
        acq_cls = _instantiate_from_target(OmegaConf.to_container(acq_cfg, resolve=True).get("_target_"))
        acq_fn = acq_cls()

    # Parse dask_client
    dask_client = None
    if OmegaConf.select(optimizer_cfg, "smac_cfg.smac_kwargs.dask_client") is not None:
        dask_client = OmegaConf.select(optimizer_cfg, "smac_cfg.smac_kwargs.dask_client")

    # Build intensifier with max_config_calls=1.
    # dacboenv always uses deterministic=True with no instances, so each config
    # has exactly one valid (instance=None, seed) key. The default max_config_calls=3
    # re-queues every config with doubled N (→2, →4), consuming ask() calls that
    # return 0 trials and contribute nothing. Setting 1 eliminates this churn.
    intensifier = facade_class.get_intensifier(scenario, max_config_calls=1)

    # Build facade
    return facade_class(
        scenario=scenario,
        target_function=target_fn,
        initial_design=initial_design,
        random_design=random_design,
        acquisition_function=acq_fn,
        intensifier=intensifier,
        overwrite=True,
        dask_client=dask_client,
    )



# Deprecated alias for backward compatibility
def build_carps_optimizer(
    task_id: str,
    seed: int,
    optimizer_id: str | None = None,
    optimizer_cfg: DictConfig | None = None,
    n_trials: int = 77,
    **kwargs,
) -> AbstractFacade:
    """Deprecated: use build_smac_facade instead."""
    return build_smac_facade(
        task_id=task_id,
        seed=seed,
        n_trials=n_trials,
        optimizer_cfg=optimizer_cfg,
    )
