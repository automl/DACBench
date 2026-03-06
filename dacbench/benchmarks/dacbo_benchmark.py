"""DACBOEnv Benchmark."""

from __future__ import annotations

from importlib.resources import files
from itertools import product
from pathlib import Path

import dacboenv
import numpy as np
import yaml
from dacboenv.env.action import AcqParameterActionSpace
from omegaconf import OmegaConf

from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs.dacbo import DACBOEnv


def load_default_optimizer():
    """Handles dacboenv configs to configure WEI as default."""
    dacboenv_path = files("dacboenv")
    base = OmegaConf.load(dacboenv_path / "configs/env/opt/base.yaml")
    base.dacboenv.optimizer_cfg.smac_cfg.smac_kwargs = None
    override = OmegaConf.load(
        dacboenv_path / "configs/env/action/wei_alpha_continuous.yaml"
    )
    cfg = OmegaConf.merge(base, override)
    cfg = OmegaConf.create({"optimizer": cfg.dacboenv.optimizer_cfg})

    def replace_refs(node):
        if isinstance(node, str):
            return node.replace("dacboenv.optimizer_cfg", "optimizer")
        if isinstance(node, dict):
            return {k: replace_refs(v) for k, v in node.items()}
        if isinstance(node, list):
            return [replace_refs(v) for v in node]
        return node

    cfg = OmegaConf.create(replace_refs(OmegaConf.to_container(cfg, resolve=False)))
    cfg.outdir = "runs/SMAC-DACBO/${benchmark_id}/${task_id}/${seed}"

    return cfg


INFO = {
    "identifier": "DACBO",
    "name": "DACBO",
    "reward": f"""Default: [symlogregret]. Other options: {[
        rew.name for rew in dacboenv.env.reward.ALL_REWARDS
    ]}""",
    "state_description": f"""Default: {[
            "ubr_difference",
            "acq_value_EI",
            "acq_value_PI",
            "previous_param",
        ]}. Other options: {[
        obs.name for obs in dacboenv.env.observation.ALL_OBSERVATIONS
    ]}""",
}

DACBO_DEFAULTS = objdict(
    {
        "reward_range": [-np.inf, np.inf],
        "seed": 0,
        "instance_set_path": "bbob_2_default.yaml",
        "optimizer_cfg": load_default_optimizer(),
        "observation_keys": [
            "ubr_difference",
            "acq_value_EI",
            "acq_value_PI",
            "previous_param",
        ],
        "action_space_class": AcqParameterActionSpace,
        "action_space_kwargs": {"bounds": [0, 1], "adjustment_type": "continuous"},
        "reward_keys": ["symlogregret"],
        "benchmark_info": INFO,
    }
)


class DACBOBenchmark(AbstractBenchmark):
    """DACBOEnv benchmark."""

    def __init__(self, config_path=None, config=None):
        """Init DACBOEnv benchmark."""
        super().__init__(config_path, config)

        if not self.config:
            self.config = objdict(DACBO_DEFAULTS.copy())

        for key in DACBO_DEFAULTS:
            if key not in self.config:
                self.config[key] = DACBO_DEFAULTS[key]

    def get_environment(self):
        """Returns the internal env."""
        if "instance_set" not in self.config:
            self.read_instance_set()

        return DACBOEnv(self.config)

    def read_instance_set(self):
        """Reads the instance set."""
        assert self.config.instance_set_path
        try:  # Look in hydra search path if user uses hydra
            from hydra.core.hydra_config import HydraConfig
            config = HydraConfig.get()
            hydra_candidate_paths = [
                Path(path_description["path"]) / self.config.instance_set_path
                for path_description in config["runtime"]["config_sources"]
                if path_description["schema"] == "file"
            ]
            matched_hydra_files = list(filter(lambda f: f.is_file(), hydra_candidate_paths))
        except ImportError:
            matched_hydra_files = []
        if Path(self.config.instance_set_path).is_file():
            path = self.config.instance_set_path
        elif len(matched_hydra_files) > 0:
            path = matched_hydra_files[0]
        else:
            path = (
                Path(__file__).resolve().parent
                / "../instance_sets/dacbo/"
                / self.config.instance_set_path
            )

        with open(path) as f:
            instance_data = yaml.safe_load(f)
        print(instance_data)
        self.config["task_ids"] = instance_data["task_ids"]
        self.config["inner_seeds"] = instance_data.get("inner_seeds", None)
        self.config["instance_set"] = dict(
            enumerate(
                product(
                    instance_data.get("inner_seeds", None), instance_data["task_ids"]
                )
            )
        )  # Not used. Instance selection is handled by the internal env

        assert len(self.config["instance_set"]) > 0, "ERROR: empty instance set"
