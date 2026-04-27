"""RL Environment for DACBO."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    SupportsFloat,
)

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, MultiDiscrete

from dacbench.envs.dacboenv.env.action import (
    AbstractActionSpace,
    AcqParameterActionSpace,
    WEITempoRLActionSpace,
)
from dacbench.envs.dacboenv.env.instance import (
    InstanceSelector,
    RoundRobinInstanceSelector,
)
from dacbench.envs.dacboenv.env.observation import ObservationSpace
from dacbench.envs.dacboenv.env.reward import DACBOReward
from dacbench.envs.dacboenv.utils.carps_optimizer import build_smac_facade
from dacbench.envs.dacboenv.utils.loggingutils import get_logger
from dacbench.envs.dacboenv.utils.math import safe_log10
from dacbench.envs.dacboenv.utils.reference_performance import ReferencePerformance

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from smac.facade.abstract_facade import AbstractFacade
    from smac.main.smbo import SMBO

    from dacbench.envs.dacboenv.env.observations.types import ObsType

ActType = int | float | list[float] | None

logger = get_logger("dacboenv")


@dataclass(frozen=True)
class InstanceSet:
    """Instance Set."""

    task_ids: list[str]
    seeds: list[int]


class DACBOEnv(gym.Env):
    """Gymnasium environment for Dynamic Algorithm Configuration in Bayesian Optimization (DACBO).

    This environment wraps a SMAC optimizer and offers a reinforcement learning interface for
    dynamically adjusting acquisition functions / parameters during Bayesian optimization.

    Observation Space
    ----------
    incumbent_changes : int
        Number of times the incumbent solution has changed.
    trials_passed : int
        Number of optimization trials completed.
    trials_left : int
        Number of trials remaining.
    ubr : float
        Upper bound regret.
    modelfit_mse : float
        Model fit measured as mean squared error.

    Action Space
    ----------
    acquisition_function : int
        Discrete selection among EI, PI, UCB, WEI.
    ei_pi_xi : float
        Parameter for EI/PI acquisition functions.
    ucb_beta : float
        Parameter for UCB acquisition function (log scale).
    wei_alpha : float
        Parameter for WEI acquisition function.

    Methods:
    -------
    step(action)
        Executes one optimization step using the selected acquisition function and parameters.
    reset(seed=None, options=None)
        Resets the environment and optimizer state.
    update_optimizer(action)
        Updates the SMAC optimizer with the given action.
    get_observation()
        Computes the current observation and reward from the optimizer.
    get_reward()
        Computes the current reward from the optimizer.
    """

    def __init__(
        self,
        task_ids: list[str],
        optimizer_cfg: DictConfig | None = None,
        observation_keys: list[str] | None = None,
        action_space_class: type[AbstractActionSpace] = AcqParameterActionSpace,
        action_space_kwargs: dict[str, Any] | None = None,
        reward_keys: list[str] | None = None,
        rho: float = 0.05,
        seed: int | None = None,
        reference_performance_fn: str = "reference_performance/reference_performance.parquet",
        reference_performance_optimizer_id: str = "SMAC3-BlackBoxFacade",
        inner_seeds: list[int] | None = None,
        terminate_after_reference_performance_reached: bool = False,
        instance_selector_class: type[InstanceSelector] | None = None,
        evaluation_mode: bool = False,
        n_trials: int = 77,
        **kwargs: Any,
    ) -> None:
        """Initialize the DACBOEnv environment.

        Parameters
        ----------
        task_ids : list[str], optional
            The task ids that BO should run on.
        optimizer_cfg : DictConfig, optional
            The SMAC optimizer config. Defaults to `SMAC3-BlackBoxFacade` which is the standard blackbox
            facade with a GP.
        observation_keys : list[str], optional
            Which observations to compute at each step.
        action_space_class : type[AbstractActionSpace], optional
            Which action space, either parameter control or acquisition function selection.
        action_space_kwargs : dict[str, Any], optional
            Keyword arguments for the action space class.
        reward_keys : list[str], optional
            Which rewards to compute at each step. If nothing provided, will be `incumbent_cost`. Beware,
            this might not make sense for DAC as the tasks live on different scales.
        rho : float, optional
            ParEGO scalarization parameter.
        inner_seeds : list[int], optional
            The seeds that the inner BO will run on.
        terminate_after_reference_performance_reached : bool, optional
            Terminate episode after a certain reference performance on a task/seed has been reached. Defaults to False.
        evaluation_mode : bool, optional
            Whether to be in train (default) or evaluation mode. Evaluation mode means that the episode is not
            terminated after a reference performance has been reached, and the reward will be 0.
            This circumvents running a reference optimizer on each evaluation task.
        n_trials : int, optional
            Maximum number of optimization trials. Defaults to 77.
        """
        if reward_keys is None:
            reward_keys = ["incumbent_cost"]
        if action_space_kwargs is None:
            action_space_kwargs = {
                # SMAC's default acquisition function is EI, thus we adjust xi, thus those are sensible default bounds
                "bounds": (-10, 10)
            }
        super().__init__()

        self._seed = seed
        # Create seed generator for resetting for new episodes
        self._seeder = np.random.default_rng(self._seed)
        self._fallback_seeds = list(self._seeder.integers(low=344, high=46483, size=3))

        self._optimizer_cfg = optimizer_cfg
        self._n_trials = n_trials
        self._action_space_class = action_space_class
        self._action_space_kwargs = action_space_kwargs
        self._action_space: AbstractActionSpace
        self._observation_keys = observation_keys
        self._reward_keys = reward_keys
        self._rho = rho

        # Instance Set
        self._instance_set: InstanceSet
        self._instance_selector_class = (
            instance_selector_class
            if instance_selector_class
            else RoundRobinInstanceSelector
        )
        self.instance_selector: InstanceSelector  # Set whenever task_id or inner_seeds are updated
        inner_seeds = inner_seeds or self._fallback_seeds
        self.instance_set = (inner_seeds, task_ids)  # type: ignore[assignment]
        self._instance: tuple[int, str] | None = None

        self._evaluation_mode = evaluation_mode
        if self._evaluation_mode:
            logger.info(
                "Env is in evaluation mode! This means that a reward is not calculated, and episodes will be full "
                "length."
            )

        # Reference Performance
        self._terminate_after_reference_performance_reached = (
            terminate_after_reference_performance_reached
        )
        if self._evaluation_mode:
            self._terminate_after_reference_performance_reached = False
        self.reference_performance_fn = reference_performance_fn
        self.reference_performance_optimizer_id = reference_performance_optimizer_id
        if not self._evaluation_mode:
            self._reference_performance = ReferencePerformance(
                optimizer_id=self.reference_performance_optimizer_id,
                task_ids=self.instance_set.task_ids,
                seeds=self.instance_set.seeds,
                reference_performance_fn=self.reference_performance_fn,
                n_trials=self._n_trials,
            )

        self._smac_facade: AbstractFacade
        self._smac_instance: SMBO

        self._episode_reward = 0.0
        self._episode_length = 0

        self.current_task_id = ""
        self.current_seed = -1
        self.current_threshold: float | None = None
        self.last_action: ActType | None = None

    @property
    def instance_set(self) -> InstanceSet:
        """The instance set."""
        return self._instance_set

    @instance_set.setter
    def instance_set(self, seeds_taskids: tuple[list[int], list[str]]) -> None:
        seeds, task_ids = seeds_taskids
        self._instance_set = InstanceSet(task_ids=task_ids, seeds=seeds)
        self._build_instance_selector()
        self._instance = None

    @property
    def instance(self) -> tuple[int, str]:
        """The intance (seed, task_ids).

        Raise
        -----
        ValueError
            When the env has to been reset after setting a new instance set.
        """
        if self._instance is None:
            raise ValueError("Reset the env first to select an instance!")
        return self._instance

    @instance.setter
    def instance(self, instance: tuple[int, str]) -> None:
        self._instance = instance

    def _build_instance_selector(self) -> None:
        self.instance_selector = self._instance_selector_class(  # type: ignore[operator]
            task_ids=self._instance_set.task_ids,
            seeds=self._instance_set.seeds,
            selector_seed=self._seed,
        )

    def update_optimizer(self, action: ActType) -> None:
        """Update the SMAC optimizer with the given action.

        Parameters
        ----------
        action : ActType
            Action specifying either the acquisition function or its parameter.

        Raises:
        ------
        ValueError
            If the action type is invalid.
        """
        if action is not None:
            self._action_space.update_optimizer(action)
            self.last_action = action

    def modify_obs(self, obs: ObsType) -> ObsType:
        """Modify observations.

        Only modify the `previous_param` observation such that it is never None.
        That would not be liked by any neural network.
        `previous_param` will be set to a default, which is the middle of the action space.

        Parameters
        ----------
        obs : ObsType
            The observations.

        Returns:
        -------
        ObsType
            The modified observations.
        """
        if "previous_param" in obs:
            if self.last_action is not None:
                previous_param = self.last_action
                if isinstance(self._action_space, WEITempoRLActionSpace):
                    assert isinstance(self.last_action, Sequence)
                    previous_param = np.array(
                        [self._action_space._param_levels[int(self.last_action[1])]]
                    )
            elif isinstance(self.action_space, Box):
                # TODO adjust default/initial action. Right now: middle of action space
                previous_param = (self.action_space.high - self.action_space.low) / 2
            elif isinstance(self._action_space, WEITempoRLActionSpace):
                assert isinstance(self.action_space, MultiDiscrete)
                n_levels = self.action_space[1].n
                previous_param = np.array(
                    [self._action_space._param_levels[n_levels // 2]]
                )
            else:
                raise ValueError(
                    f"Cannot handle space {self.action_space} to set last action."
                )

            obs["previous_param"] = previous_param
        return obs

    def get_observation(self) -> ObsType:
        """Compute the current observation from the optimizer.

        Returns:
        -------
        obs : dict[str, Any]
            Dictionary of observation values.
        """
        obs = self._dacbo_observation_space.get_observation()
        return self.modify_obs(obs=obs)

    def get_reward(self) -> float:
        """Compute the current reward from the optimizer.

        Returns:
        -------
        reward : float
            The current reward signal.
        """
        if not self._evaluation_mode:
            return self._reward.get_reward(self.current_threshold)
        return 0

    def get_next_instance(self) -> tuple[int, str]:
        """Get the next instance.

        Returns:
        -------
        tuple[int,str]
            (seed,task_id)
        """
        return self.instance_selector.select_instance()  # type: ignore[return-value]

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one optimization step using the selected acquisition function and parameters.

        Parameters
        ----------
        action : ActType
            Action specifying either the acquisition function or its parameter.

        Returns:
        -------
        obs : dict
            The new observation after taking the action.
        reward : float
            The reward for the action taken.
        terminated : bool
            Whether the episode has terminated (reference performance reached).
        truncated : bool
            Whether the episode was truncated (always False).
        info : dict
            Additional information (empty).
        """
        if isinstance(self._action_space, WEITempoRLActionSpace):
            assert isinstance(action, Sequence)
            step_duration = self._action_space._step_durations[int(action[0])]
            param_level = action[1]
            logger.info(f"Do action {param_level} for {step_duration} steps.")
            for _i in range(step_duration):
                # TODO Fix RL training logging for this as this seems that the episode length is way shorter
                obs = self._step(action=action)
            return obs
        return self._step(action=action)

    def _step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one optimization step using the selected acquisition function and parameters.

        Parameters
        ----------
        action : ActType
            Action specifying either the acquisition function or its parameter.

        Returns:
        -------
        obs : dict
            The new observation after taking the action.
        reward : float
            The reward for the action taken.
        terminated : bool
            Whether the episode has terminated (reference performance reached).
        truncated : bool
            Whether the episode was truncated (always False).
        info : dict
            Additional information (empty).
        """
        self.update_optimizer(action)

        # BO step
        trial_info = self._smac_instance.ask()
        _, trial_value = self._smac_instance._runner.run_wrapper(trial_info)
        self._smac_instance.tell(trial_info, trial_value)

        terminated = False

        curr_incumbent = self.get_incumbent_cost()
        if not self._evaluation_mode:
            threshold = self._reference_performance.query_cost(  # type: ignore[attr-defined]
                optimizer_id=self.reference_performance_optimizer_id,
                task_id=self.current_task_id,
                seed=self.current_seed,
            )
            self.current_threshold = threshold

        if self._terminate_after_reference_performance_reached:
            distance = abs(curr_incumbent - threshold)
            log_distance = safe_log10(distance)
            logger.info(
                f"Current: {curr_incumbent:.4f}, threshold: {threshold:.4f}, log distance: {log_distance:.4f}"
            )
            terminated = curr_incumbent < threshold  # We minimize

        remaining_trials = (
            self._smac_instance._scenario.n_trials
            - self._smac_instance.runhistory.finished
        )
        truncated = remaining_trials <= 0

        # Compute observation + reward
        obs = self.get_observation()
        reward = self.get_reward() if not self._evaluation_mode else 0

        self._episode_reward += reward
        self._episode_length += 1

        info = {}
        if terminated or truncated:
            info["episode"] = {"r": self._episode_reward, "l": self._episode_length}
            self._episode_reward = 0
            self._episode_length = 0

        logger.info(
            f"Step: {self._episode_length}, instance: {self.instance}, action: {action}, reward: {reward}, "
            f"terminated: {terminated}, truncated: {truncated}, info: {info}"
        )

        return obs, reward, terminated, truncated, info

    def get_incumbent_cost(self) -> float:
        """Get the current incumbent cost.

        Returns:
        -------
        float
            Minimum cost found so far on this target function (not necessarily the reward).
        """
        return self._smac_instance.runhistory.get_min_cost(
            self._smac_instance.intensifier.get_incumbent()
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict, optional
            Additional reset options.

        Returns:
        -------
        obs : tuple
            The initial observation.
        info : dict
            Additional information (empty).
        """
        # Reset SMAC instance
        if hasattr(self, "_smac_instance"):
            del self._smac_instance

        # Get next instance which is a combo of task id and seed
        self.instance = self.get_next_instance()
        seed, task_id = self.instance
        if seed is None:
            seed = self._seeder.integers(low=0, high=2**32 - 1)
        seed = int(seed)

        # Build SMAC facade directly
        self._smac_facade = build_smac_facade(
            task_id=task_id,
            seed=seed,
            n_trials=self._n_trials,
            optimizer_cfg=self._optimizer_cfg,
        )
        self._smac_instance = self._smac_facade.optimizer

        if self._smac_instance._scenario.count_objectives() != 1:
            raise NotImplementedError("Multi-objective not supported.")

        # Setup observation space
        self._dacbo_observation_space = ObservationSpace(
            self._smac_instance, self._observation_keys
        )
        self._dacbo_observation_space.reset()
        self.observation_space = (
            self._dacbo_observation_space.space
        )  # gym observation space

        # Setup action space
        self._action_space = self._action_space_class(
            smac_instance=self._smac_instance, **self._action_space_kwargs
        )
        self.action_space = self._action_space.space  # gym action space
        self.action_space.seed(seed)  # Seed with current seed
        self.last_action = None

        # If previous_param is in obs, define the observation space for it
        if "previous_param" in self._dacbo_observation_space._keys:  # type: ignore
            self._dacbo_observation_space._observation_space[
                "previous_param"
            ] = self.action_space
            if isinstance(self._action_space, WEITempoRLActionSpace):
                self._dacbo_observation_space._observation_space[
                    "previous_param"
                ] = Box(low=0, high=1)

        # Setup reward
        self._reward = DACBOReward(self._smac_instance, self._reward_keys, self._rho)

        super().reset(seed=seed)
        self.current_seed = seed
        self.current_task_id = task_id

        if not self._evaluation_mode:
            # Work off new initial design
            # This is important for training DAC policies because for the phase of the initial design, no action can
            # be taken and this might lead to misleading signals.
            # In evaluation, however, the initial design counts towards the total number of trials, controlled by
            # SMAC optimizer.
            for _ in (
                self._smac_instance.intensifier.config_selector._initial_design_configs
            ):
                trial_info = self._smac_instance.ask()
                _, trial_value = self._smac_instance._runner.run_wrapper(trial_info)
                self._smac_instance.tell(trial_info, trial_value)

        initial_obs = {
            obs.name: np.atleast_1d(obs.default).astype(np.float32)
            for obs in self._dacbo_observation_space._observation_types
        }
        initial_obs = self.modify_obs(obs=initial_obs)

        return initial_obs, {}

    def get_n_finished_trials(self) -> int:
        """Get the number of told trials from the SMAC instance.

        Returns:
        -------
        int
            Number of observations
        """
        return self._smac_instance._runhistory._finished
