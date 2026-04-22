"""Build carps optimizer."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from carps.utils.env_vars import CARPS_ROOT
from carps.utils.running import make_optimizer, make_task

with contextlib.suppress(Exception):
    from carps.utils.index_configs import get_index
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from carps.optimizers.optimizer import Optimizer
    from omegaconf import DictConfig


def load_optimizer_config(optimizer_id: str) -> DictConfig:
    """Load optimizer config from yaml file.

    The config can also have defaults=["base"], but not any other defaults structure.

    Parameters
    ----------
    optimizer_id : str
        carps optimizer_id or the filename of the optimizer config (yaml).

    Returns:
    -------
    DictConfig
        The optimizer config.
    """
    if optimizer_id.endswith(".yaml"):
        config_fn = optimizer_id
    else:
        index_fn = CARPS_ROOT / "configs/optimizer/index.csv"
        try:
            df = get_index()
        except NameError:
            df = pd.read_csv(index_fn)
        ids = [optimizer_id]
    config_fn = df.set_index("optimizer_id").loc[ids].reset_index().iloc[0]["config_fn"]
    cfg = OmegaConf.load(config_fn)
    return maybe_add_defaults(cfg, config_fn)


def maybe_add_defaults(cfg: DictConfig, cfg_fn: str) -> DictConfig:
    """Maybe add default config to config.

    Only works with defaults = ["base"].

    Parameters
    ----------
    cfg : DictConfig
        The config.
    cfg_fn : str
        The source config filename.

    Returns:
    -------
    DictConfig
        Cfg, possibly with defaults added.

    Raises:
    ------
    ValueError
        When got other defaults than ['base'].
    """
    defaults = cfg.get("defaults", None)
    if defaults is not None:
        if list(cfg.defaults) == ["base"]:
            cfg = OmegaConf.merge(
                cfg, OmegaConf.load(Path(cfg_fn).parent / "base.yaml")
            )
            del cfg.defaults
        else:
            raise ValueError(
                f"Can only handle defaults=['base'], but got {cfg.defaults}"
            )
    return cfg


def get_task_config(task_id: str) -> DictConfig:
    """Get config filename for task id.

    Parameters
    ----------
    task_id : str
        The task id.

    Returns:
    -------
    DictConfig
        The config with the node task.
    """
    task_index_fn = CARPS_ROOT / "configs/task/index.csv"
    try:
        df = get_index()
    except NameError:
        df = pd.read_csv(task_index_fn)

    ids = [task_id]
    # TODO raise proper error if task_id not in index. Can happen when task comes from external module.
    # Find smart registering method.
    config_fn = df.set_index("task_id").loc[ids].reset_index().iloc[0]["config_fn"]
    cfg = OmegaConf.load(config_fn)
    return maybe_add_defaults(cfg, config_fn)


def build_carps_optimizer(
    task_id: str,
    seed: int,
    optimizer_id: str | None = None,
    optimizer_cfg: DictConfig | None = None,
) -> Optimizer:
    """Build carps optimizer.

    Later, the built SMAC solver can be used.
    Either specify `optimizer_id` or `optimizer_cfg`.

    Parameters
    ----------
    task_id : str
        The carps task id.
    seed : int
        The seed.
    optimizer_id : str, optional
        The carps optimizer id.
    optimizer_cfg : DictConfig, optional
        The optimizer config.

    Returns:
    -------
    Optimizer
        carps optimizer.
    """
    if optimizer_id is None and optimizer_cfg is None:
        raise ValueError("Specify either optimizer_id or optimizer_cfg!")

    cfg_opt = optimizer_cfg or None
    if cfg_opt is None:
        cfg_opt = load_optimizer_config(optimizer_id=optimizer_id)  # type: ignore[arg-type]

    cfg = get_task_config(task_id=task_id)
    cfg.seed = seed

    if hasattr(cfg_opt, "optimizer"):
        cfg = OmegaConf.merge(cfg, cfg_opt)
    else:
        cfg.optimizer = cfg_opt

    if not hasattr(cfg.optimizer, "_target_"):
        cfg.optimizer._target_ = "carps.optimizers.smac20.SMAC3Optimizer"
        cfg.optimizer._partial_ = True

    if hasattr(cfg, "loggers"):
        del cfg.loggers
    task = make_task(cfg=cfg)

    if OmegaConf.select(cfg, "optimizer.smac_cfg.scenario.n_trials") is not None:
        cfg.optimizer.smac_cfg.scenario.n_trials = (
            cfg.task.optimization_resources.n_trials
        )

    optimizer = make_optimizer(cfg=cfg, task=task)
    optimizer.setup_optimizer()
    return optimizer
