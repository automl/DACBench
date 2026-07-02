"""Reward utilities for DACBOEnv."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from sklearn.metrics import auc

from dacbench.envs.dacboenv.utils.math import symlog
from dacbench.envs.dacboenv.utils.parego import ParEGO

if TYPE_CHECKING:
    from smac.main.smbo import SMBO


@dataclass
class RewardType:
    """Represents a single reward type for the DACBO environment.

    Attributes:
    ----------
    name : str
        Name of the reward.
    compute : Callable[[SMBO], Any]
        Function to compute the reward value from a SMAC instance and from
        reference_performance: float | None.
    """

    name: str
    compute: Callable[[SMBO, float | None], Any]


# Multi-objective: Handle incumbent cost

auc_reward = RewardType(
    "trajectory_auc",
    lambda smbo, reference_performance: -auc(
        [t.trial for t in smbo.intensifier.trajectory], costs
    )
    if len(
        costs := [
            t.costs[-1] - smbo.intensifier.trajectory[0].costs[-1]
            for t in smbo.intensifier.trajectory
        ]
    )
    > 1
    else 0,
)
incumbent_cost_reward = RewardType(
    "incumbent_cost",
    lambda smbo, reference_performance: -smbo.intensifier.trajectory[-1].costs[-1],
)  # Minimize cost
incumbent_improvement_reward = RewardType(
    "incumbent_improvement",
    lambda smbo, reference_performance: abs(
        smbo.intensifier.trajectory[-1].costs[-1]
        - smbo.intensifier.trajectory[-2].costs[-1]
    )
    if len(smbo.intensifier.trajectory) > 1
    and smbo.intensifier.trajectory[-1].trial == len(smbo.runhistory)
    else 0,
)
sqrt_incumbent_improvement_reward = RewardType(
    "sqrt_incumbent_improvement",
    lambda smbo, reference_performance: np.sqrt(
        abs(
            smbo.intensifier.trajectory[-1].costs[-1]
            - smbo.intensifier.trajectory[-2].costs[-1]
        )
    )
    if len(smbo.intensifier.trajectory) > 1
    and smbo.intensifier.trajectory[-1].trial == len(smbo.runhistory)
    else 0,
)
auc_reward_alt = RewardType(
    "trajectory_auc_alt",
    lambda smbo, reference_performance: -auc(
        range(len(smbo.runhistory)),
        np.minimum.accumulate(
            [
                t.cost - smbo.intensifier.trajectory[0].costs[-1]
                for t in smbo.runhistory.values()
            ]
        ),
    )
    if len(smbo.runhistory) > 1
    else 0,
)


def get_initial_design_size(solver: SMBO) -> int:
    """Get the size of the initial design.

    Parameters
    ----------
    solver : smac.main.smbo.SMBO
        The optimizer.

    Returns:
    -------
    int
        Initial design size.
    """
    return len(solver.intensifier.config_selector._initial_design_configs)


def get_reward_for_episode_finished(
    smbo: SMBO,
    reference_performance: float | None = None,
    scale_by_budget: bool = False,
) -> float:
    """Get reward (or rather punishment: -1) as long the episode is not finished.

    Typically, the episode is finished after DACBO has reached reference performance.

    Parameters
    ----------
    smbo : SMBO
        The SMAC instance.
    scale_by_budget : bool, optional
        Whether to scale by the model-based budget, by default False. If yes, return -1/b.

    Returns:
    -------
    float
        Reward value: -1 if the episode is not finished, or -1 divided by the model-based budget
        if `scale_by_budget` is True.
    """
    if not scale_by_budget:
        return -1

    n_initial_design = get_initial_design_size(smbo)
    n_smbo = smbo._scenario.n_trials
    n_model_based = n_smbo - n_initial_design

    return -1 / n_model_based


episode_finished = RewardType("episode_finished", get_reward_for_episode_finished)

episode_finished_scaled = RewardType(
    "episode_finished_scaled",
    partial(get_reward_for_episode_finished, scale_by_budget=True),
)


def calc_symlogregret_of_reference_performance(
    smbo: SMBO, reference_performance: float | None = None
) -> float:
    """Calculate the symmetric log regret to the reference performance.

    Parameters
    ----------
    smbo : SMBO
        The SMAC instance.
    reference_performance : float | None, optional
        The reference performance., by default None

    Returns:
    -------
    float
        The symlog regret.
    """
    cost_inc = smbo.runhistory.get_min_cost(smbo.intensifier.get_incumbent())
    diff = reference_performance - cost_inc
    return symlog(diff)


symlogregret_reward = RewardType(
    "symlogregret", calc_symlogregret_of_reference_performance
)

ALL_REWARDS = [
    auc_reward,
    incumbent_cost_reward,
    incumbent_improvement_reward,
    sqrt_incumbent_improvement_reward,
    auc_reward_alt,
    episode_finished,
    episode_finished_scaled,
    symlogregret_reward,
]


class DACBOReward:
    """Manages a collection of reward types and computes (possibly multi-objective) rewards.

    Supports scalarization of multiple reward objectives using ParEGO.

    Parameters
    ----------
    smac_instance : SMBO
        The SMAC optimizer instance.
    keys : list[str], optional
        List of reward names to include. If None, all available rewards are used.
    rho : float, optional
        ParEGO scalarization parameter (default: 0.05).

    Attributes:
    ----------
    _reward_types : list[RewardType]
        The selected reward types.
    _parego : ParEGO
        ParEGO scalarization utility.

    Methods:
    -------
    get_reward() -> float
        Computes the (scalarized) reward from the selected reward types.
    """

    _REWARD_MAP: ClassVar[dict[str, RewardType]] = {
        rew.name: rew for rew in ALL_REWARDS
    }

    def __init__(
        self, smac_instance: SMBO, keys: list[str] | None = None, rho: float = 0.05
    ) -> None:
        """Initialize the DACBOReward.

        Parameters
        ----------
        smac_instance : SMBO
            The SMAC optimizer instance.
        keys : list[str], optional
            List of reward names to include. If None, all available rewards are used.
        rho : float, optional
            ParEGO scalarization parameter (default: 0.05).

        Raises:
        ------
        ValueError
            If any provided keys are not valid reward names.
        """
        self._smac_instance = smac_instance
        self._rho = rho

        # Default to all possible keys if not provided
        self._keys = keys if keys is not None else list(self._REWARD_MAP.keys())

        # Check for invalid keys
        invalid_keys = set(self._keys) - set(self._REWARD_MAP.keys())
        if invalid_keys:
            raise ValueError(f"Invalid reward keys: {invalid_keys}")

        self._reward_types = [self._REWARD_MAP[key] for key in self._keys]

        self._parego = ParEGO(
            len(self._reward_types), self._smac_instance._scenario.seed, self._rho
        )

    def _get_full_reward(
        self, reference_performance: float | None = None
    ) -> dict[str, float]:
        """Compute all sub-rewards from the selected reward types.

        Returns:
        -------
        dict[str, float]
            All sub-rewards.
        """
        return {
            rew.name: rew.compute(self._smac_instance, reference_performance)
            for rew in self._reward_types
        }

    def get_reward(self, reference_performance: float | None = None) -> float:
        """Compute the (scalarized) reward from the selected reward types.

        Returns:
        -------
        float
            The computed reward value.
        """
        full_reward = self._get_full_reward(reference_performance=reference_performance)
        if len(self._reward_types) == 1:
            return next(iter(full_reward.values()))
        # Multi-objective using ParEGO
        return self._parego(list(full_reward.values()))
