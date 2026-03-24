"""Observation utilities for DACBOEnv."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
)

import numpy as np
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
    OrdinalHyperparameter,
)
from gymnasium.spaces import Box, Dict, Space
from scipy.stats import kurtosis, skew
from smac.main.smbo import SMBO

from dacbench.envs.dacboenv.env.observations.acquisition_function import (
    GetAFandAcqValue,
    acq_value_ei_observation,
    acq_value_pi_observation,
    acq_value_wei_explore_observation,
    acq_value_wei_observation,
)
from dacbench.envs.dacboenv.env.observations.types import (
    MultiObservationType,
    ObservationType,
    ObsType,
)
from dacbench.envs.dacboenv.features.signal.modelfit import calculate_model_fit
from dacbench.envs.dacboenv.features.signal.ubr import calculate_ubr
from dacbench.envs.dacboenv.features.X_features import exploration_tsd, knn_entropy
from dacbench.envs.dacboenv.features.y_features import calc_variability
from dacbench.envs.dacboenv.policy.sawei import apply_moving_iqm

if TYPE_CHECKING:
    from smac.main.smbo import SMBO

    from dacbench.envs.dacboenv.env.observations.types import Memory, ObsType


def get_best_percentile_configs(
    smbo: SMBO,
    p: int = 10,
    min_samples: int = 1,
    memory: Memory | None = None,
) -> np.ndarray:
    """Returns the best 1/p percent of configs."""
    configs_sorted = [
        k.config_id
        for k, _ in sorted(smbo.runhistory._data.items(), key=lambda x: x[1].cost)
    ]
    n = max(min_samples, len(configs_sorted) // p)
    return np.array(
        [
            smbo.runhistory.get_config(config_id).get_array()
            for config_id in configs_sorted[:n]
        ]
    )


def get_best_percentile_costs(
    smbo: SMBO,
    p: int = 10,
    min_samples: int = 1,
    memory: Memory | None = None,
) -> np.ndarray:
    """Returns the best 1/p percent of costs."""
    costs_sorted = [
        v.cost
        for _, v in sorted(smbo.runhistory._data.items(), key=lambda x: x[1].cost)
    ]
    n = max(min_samples, len(costs_sorted) // p)
    return np.array(costs_sorted[:n])


def enumerate_offset(hyperparameters: Sequence[Any]) -> Iterator[tuple[int, Any]]:
    """Enumerates the given hyperparameters along with their running length as offset."""
    offset = 0
    for hp in hyperparameters:
        yield offset, hp
        offset += hp.n_elements


def calc_last_diff(memory: Memory, key: str) -> float:
    """Calc the last difference in a signal.

    Parameters
    ----------
    memory : Memory
        The memory/history of state features.
    key : str
        The name of the observation.

    Returns:
    -------
    float
        The last diff.
    """
    return 0 if len(memory[key]) < 2 else memory[key][-2] - memory[key][-1]


class ComputeUBR:
    """Compute the UBR.

    Currently no need to have it as a class.
    """

    def __call__(self, smbo: SMBO) -> float | None:
        """Compute the UBR.

        Parameters
        ----------
        smbo : SMBO
            The SMAC instance.

        Returns:
        -------
        float
            The current UBR.
        """
        result_dict = calculate_ubr(
            trial_infos=None, trial_values=None, configspace=None, smbo=smbo
        )
        return result_dict.get("ubr", 0)


def get_last_val(memory: Memory, key: str) -> float:
    """Get the last value of a signal in memory.

    Parameters
    ----------
    memory : Memory
        The memory/history of state features.
    key : str
        The name of the observation.

    Returns:
    -------
    float
        The last/newest value.
    """
    return memory[key][-1]


def ubr_difference(memory: Memory) -> float:
    """Computes the difference between the last two KNN values."""
    return calc_last_diff(memory=memory, key="ubr")


def knn_difference(memory: Memory) -> float:
    """Computes the difference between the last two KNN values."""
    return calc_last_diff(memory=memory, key="knn")


incumbent_change_observation = ObservationType(
    "incumbent_changes",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo, memory: smbo.intensifier.incumbents_changed,
    0,
)
trials_passed_observation = ObservationType(
    "trials_passed",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo, memory: len(smbo.runhistory),
    0,
)
trials_left_observation = ObservationType(
    "trials_left",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo, memory: smbo.remaining_trials,
    -1,
)
ubr_observation = ObservationType(
    "ubr",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo, memory: get_last_val(memory=memory, key="ubr"),  # type: ignore[arg-type]
    -1,
)


def calc_gradient(memory: Memory, key: str, smooth_signal: bool = False) -> np.ndarray:
    """Calc the gradient of a signal in memory.

    Parameters
    ----------
    memory : Memory
        The memory/history of state features/observations.
    key : str
        The name of the observation.
    smooth_signal : bool, optional
        Whether to smooth the signal, by default False. If True, a moving IQM is applied with a window length of 7.
        This also introduces a slight delay in the signal.

    Returns:
    -------
    np.ndarray
        The gradient of a signal, possibly smoothed.
    """
    raw_signal = memory[key]
    raw_signal = [r for r in raw_signal if r is not None]
    if len(raw_signal) == 0:
        return np.array([0])
    maybe_smoothed_signal = (
        apply_moving_iqm(raw_signal, window_size=7) if smooth_signal else raw_signal
    )
    if len(maybe_smoothed_signal) == 1:
        return np.array([0])
    if len(maybe_smoothed_signal) < 3:
        return np.diff(maybe_smoothed_signal)
    return np.gradient(maybe_smoothed_signal)


def calc_ubr_gradient(memory: Memory, smooth_signal: bool = False) -> np.ndarray:
    """Calc the gradient of the UBR.

    Parameters
    ----------
    memory : Memory
        The memory/history of state features/observations.
    smooth_signal : bool, optional
        Whether to smooth the signal, by default False. If True, a moving IQM is applied with a window length of 7.
        This also introduces a slight delay in the signal.

    Returns:
    -------
    np.ndarray
        The gradient of UBR, possibly smoothed.
    """
    return calc_gradient(memory=memory, key="ubr", smooth_signal=smooth_signal)


ubr_gradient_observation = ObservationType(
    "ubr_gradient",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo, memory: calc_ubr_gradient(memory=memory, smooth_signal=False)[-1],  # type: ignore[arg-type]
    0,
)
ubr_smoothed_gradient_observation = ObservationType(
    "ubr_smoothed_gradient",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo, memory: calc_ubr_gradient(memory=memory, smooth_signal=True)[-1],  # type: ignore[arg-type]
    0,
)
ubr_smoothed_gradient_std_observation = ObservationType(
    "ubr_smoothed_gradient_std",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo, memory: np.std(calc_ubr_gradient(memory=memory, smooth_signal=True)),  # type: ignore[arg-type]
    0,
)
modelfit_observation = ObservationType(
    "modelfit_mse",
    Box(low=0.0, high=np.inf, dtype=np.float32),
    lambda smbo, memory: -1
    if np.isnan(scores := calculate_model_fit(smbo)["mean_scores"]).any()
    else scores[0],
    -1,
)
dimensions_observation = ObservationType(
    "searchspace_dim",
    Box(low=0, high=np.inf, dtype=np.int32),
    lambda smbo, memory: len(smbo._scenario.configspace),
    0,
)
continuous_hp_observation = ObservationType(
    "continuous_hps",
    Box(low=0, high=np.inf, dtype=np.int32),
    lambda smbo, memory: len(
        [
            hp
            for hp in smbo._scenario.configspace.values()
            if isinstance(hp, FloatHyperparameter)
        ]
    ),
    0,
)
categorical_hp_observation = ObservationType(
    "categorical_hps",
    Box(low=0, high=np.inf, dtype=np.int32),
    lambda smbo, memory: len(
        [
            hp
            for hp in smbo._scenario.configspace.values()
            if isinstance(hp, CategoricalHyperparameter)
        ]
    ),
    0,
)
ordinal_hp_observation = ObservationType(
    "ordinal_hps",
    Box(low=0, high=np.inf, dtype=np.int32),
    lambda smbo, memory: len(
        [
            hp
            for hp in smbo._scenario.configspace.values()
            if isinstance(hp, OrdinalHyperparameter)
        ]
    ),
    0,
)
int_hp_observation = ObservationType(
    "int_hps",
    Box(low=0, high=np.inf, dtype=np.int32),
    lambda smbo, memory: len(
        [
            hp
            for hp in smbo._scenario.configspace.values()
            if isinstance(hp, IntegerHyperparameter)
        ]
    ),
    0,
)
tsd_observation = ObservationType(
    "tsd",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo, memory: exploration_tsd(
        smbo.intensifier.config_selector._collect_data()[0]
    )[-1],
    -1,
)


def calculate_knn(smbo: SMBO) -> float:
    """Calculate KNN exploration measure following Papenmeier et al. (2025, Exploring exploration in BO).

    Parameters
    ----------
    smbo : SMBO
        The SMAC instance.

    Returns:
    -------
    float
        The KNN value.
    """
    if (
        len(configs := smbo.intensifier.config_selector._collect_data()[0]) > 3
    ):  # (default k == 3)
        return knn_entropy(configs)
    return 0


knn_entropy_observation = ObservationType(
    "knn_entropy",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo, memory: get_last_val(memory=memory, key="knn"),  # type: ignore[arg-type]
    0,
)
y_skewness_observation = ObservationType(
    "y_skewness",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo, memory: np.nan_to_num(skew(costs).item(), nan=0)
    if len(costs := smbo.intensifier.config_selector._collect_data()[1]) > 0
    else 0,
    0,
)
y_kurtosis_observation = ObservationType(
    "y_kurtosis",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo, memory: np.nan_to_num(kurtosis(costs).item(), nan=0)
    if len(costs := smbo.intensifier.config_selector._collect_data()[1]) > 0
    else 0,
    0,
)
y_mean_observation = ObservationType(
    "y_mean",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo, memory: np.mean(costs)
    if len(costs := smbo.intensifier.config_selector._collect_data()[1]) > 0
    else 0,
    0,
)
std_observation = ObservationType(
    "y_std",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo, memory: np.std(costs)
    if len(costs := smbo.intensifier.config_selector._collect_data()[1]) > 0
    else -1,
    -1,
)
variability_observation = ObservationType(
    "y_variability",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo, memory: calc_variability(costs)
    if len(costs := smbo.intensifier.config_selector._collect_data()[1]) > 3
    else -1,
    -1,
)
tsd_best_observation = ObservationType(
    "tsd_best",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo, memory: exploration_tsd(get_best_percentile_configs(smbo))[-1],
    -1,
)
knn_entropy_best_observation = ObservationType(
    "knn_entropy_best",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo, memory: knn_entropy(configs)  # type: ignore[arg-type]
    if len(configs := get_best_percentile_configs(smbo, min_samples=4))
    > 3  # (default k == 3)
    else 0,
    0,
)
skewness_best_observation = ObservationType(
    "y_skewness_best",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo, memory: np.nan_to_num(skew(costs).item(), nan=0)
    if len(costs := get_best_percentile_costs(smbo)) > 0
    else 0,
    0,
)
kurtosis_best_observation = ObservationType(
    "y_kurtosis_best",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo, memory: np.nan_to_num(kurtosis(costs).item(), nan=0)
    if len(costs := get_best_percentile_costs(smbo)) > 0
    else 0,
    0,
)
mean_best_observation = ObservationType(
    "y_mean_best",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo, memory: np.mean(costs)
    if len(costs := get_best_percentile_costs(smbo)) > 0
    else 0,
    0,
)
std_best_observation = ObservationType(
    "y_std_best",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo, memory: np.std(costs)
    if len(costs := get_best_percentile_costs(smbo)) > 0
    else -1,
    -1,
)
variability_best_observation = ObservationType(
    "y_variability_best",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo, memory: calc_variability(costs)  # type: ignore[arg-type]
    if len(costs := get_best_percentile_costs(smbo, min_samples=4)) > 3
    else -1,
    -1,
)
budget_percentage_observation = ObservationType(
    "budget_percentage",
    Box(low=0, high=1, dtype=np.float32),
    lambda smbo, memory: len(smbo.runhistory) / smbo._scenario.n_trials,
    0,
)
inc_improvement_scaled_observation = ObservationType(
    "inc_improvement_scaled",
    Box(low=0, high=1, dtype=np.float32),
    lambda smbo, memory: 1 - min(curr, prev) / max(curr, prev)
    if len(t := smbo.intensifier.trajectory) > 1
    and t[-1].trial == len(smbo.runhistory)
    and max(curr := abs(t[-1].costs[-1]), prev := abs(t[-2].costs[-1])) != 0
    else 0,
    0,
)
has_categorical_hps = ObservationType(
    "has_categorical_hps",
    Box(low=0, high=1, dtype=bool),
    lambda smbo, memory: len(
        [
            hp
            for hp in smbo._scenario.configspace.values()
            if isinstance(hp, CategoricalHyperparameter)
        ]
    )
    > 0,
    False,
)
knn_difference_observation = ObservationType(
    "knn_difference",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo, memory: knn_difference(memory=memory),  # type: ignore[arg-type]
    0,
)
ubr_difference_observation = ObservationType(
    "ubr_difference",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo, memory: ubr_difference(memory),  # type: ignore[arg-type]
    0,
)

# Must be computed INSIDE DACBOEnv, because observationspace does not have access to action space and last action
previous_param_observation = ObservationType(
    name="previous_param",
    space=Box(low=-np.inf, high=np.inf, dtype=np.float32),
    compute=lambda smbo, memory: None,
    default=None,
)


def build_gp_hp_observations(smbo: SMBO) -> list[ObservationType]:
    """Build the GP Hyperparameter Observations.

    Parameters
    ----------
    smbo : SMBO
        The SMAC instance.

    Returns:
    -------
    list[ObservationType]
        A list of the single GP HPs.
    """
    observations = []

    for offset, hp in enumerate_offset(
        smbo._intensifier._config_selector._acquisition_function.model._kernel.hyperparameters
    ):
        if hp.fixed:
            continue

        for i in range(hp.n_elements):
            idx = i + offset

            observations.append(
                ObservationType(
                    name=f"gp_hp_{hp.name}{i}_observation",
                    space=Box(hp.bounds[i][0], hp.bounds[i][1]),
                    compute=lambda smbo_, memory=None, idx=idx: (  # type: ignore[misc]
                        smbo_._intensifier._config_selector._acquisition_function.model._kernel.theta[
                            idx
                        ]
                    ),
                    default=0,
                )
            )

    return observations


gp_hp_observation = MultiObservationType(
    "gp_hp_observations",
    build_gp_hp_observations,
)

ALL_OBSERVATIONS = [
    incumbent_change_observation,
    trials_passed_observation,
    trials_left_observation,
    # modelfit_observation, # Disabled due to high computation time, behavior similar to UBR
    dimensions_observation,
    continuous_hp_observation,
    categorical_hp_observation,
    ordinal_hp_observation,
    int_hp_observation,
    tsd_observation,
    y_skewness_observation,
    y_kurtosis_observation,
    y_mean_observation,
    std_observation,
    variability_observation,
    tsd_best_observation,
    skewness_best_observation,
    kurtosis_best_observation,
    mean_best_observation,
    std_best_observation,
    variability_best_observation,
    budget_percentage_observation,
    inc_improvement_scaled_observation,
    has_categorical_hps,
    acq_value_ei_observation,
    acq_value_wei_observation,
    acq_value_pi_observation,
    acq_value_wei_explore_observation,
    previous_param_observation,
    ubr_observation,
    ubr_gradient_observation,
    ubr_smoothed_gradient_observation,
    ubr_smoothed_gradient_std_observation,
    ubr_difference_observation,
    knn_entropy_observation,
    knn_entropy_best_observation,
    knn_difference_observation,
]

MULTI_OBSERVATIONS = [gp_hp_observation]


class ObservationSpace:
    """Manages a collection of observation types and their Gymnasium spaces.

    Allows selection of a subset of available observation types and provides methods to
    compute observations from a SMAC instance.

    Parameters
    ----------
    smac_instance : SMBO
        The SMAC instance.
    keys : list[str], optional
        List of observation names to include. If None, all available observations are used.

    Attributes:
    ----------
    observation_types : list[ObservationType]
        The list containing all selected observation types.
    observation_space : gymnasium.spaces.Dict
        The Gymnasium Dict space describing the selected observations.

    Methods:
    -------
    get_observation(optimizer: SMBO) -> ObsType
        Computes the current observation values from the given optimizer.
    """

    _OBSERVATION_MAP: ClassVar[dict[str, ObservationType]] = {
        obs.name: obs for obs in ALL_OBSERVATIONS
    }
    _MULTI_OBSERVATION_MAP: ClassVar[dict[str, MultiObservationType]] = {
        obs.name: obs for obs in MULTI_OBSERVATIONS
    }

    def __init__(self, smac_instance: SMBO, keys: list[str] | None = None) -> None:
        """Initialize the ObservationSpace.

        Parameters
        ----------
        smac_instance : SMBO
            The SMAC instance.
        keys : list[str], optional
            List of observation names to include. If None, all available observations are used.

        Raises:
        ------
        ValueError
            If any provided key is invalid.
        """
        self._smac_instance = smac_instance

        # Default to all possible keys if not provided
        self._keys = (
            keys
            if keys is not None
            else list(ObservationSpace._OBSERVATION_MAP.keys())
            + list(ObservationSpace._MULTI_OBSERVATION_MAP.keys())
        )

        # Check for invalid keys
        invalid_keys = (
            set(self._keys)
            - set(ObservationSpace._OBSERVATION_MAP.keys())
            - set(ObservationSpace._MULTI_OBSERVATION_MAP.keys())
        )
        if invalid_keys:
            raise ValueError(f"Invalid observation keys: {invalid_keys}")

        self._observation_types = [
            ObservationSpace._OBSERVATION_MAP[key]
            for key in self._keys
            if key in ObservationSpace._OBSERVATION_MAP
        ] + [
            space
            for key in self._keys
            if key in ObservationSpace._MULTI_OBSERVATION_MAP
            for space in ObservationSpace._MULTI_OBSERVATION_MAP[key].create(
                smac_instance
            )
        ]
        for obs in self._observation_types:
            if inspect.isclass(obs.compute):
                obs.compute = obs.compute()
        self._observation_space = Dict(
            {obs.name: obs.space for obs in self._observation_types}
        )

        self._register_to_memory: dict[str, Callable] = {}
        self._memory: Memory = {}
        for obs in self._observation_types:
            if obs.name.startswith("ubr"):
                self._register_to_memory["ubr"] = ComputeUBR()
                self._memory["ubr"] = []
            elif obs.name.startswith("knn") and "best" not in obs.name:
                self._register_to_memory["knn"] = calculate_knn
                self._memory["knn"] = []

    @property
    def space(self) -> Space:
        """Returns the Gymnasium Dict space for the selected observations.

        Returns:
        -------
        gymnasium.spaces.Dict
            The observation space.
        """
        return self._observation_space

    def get_observation(self) -> ObsType:
        """Compute the current observation values from the given optimizer.

        Returns:
        -------
        ObsType
            Dictionary mapping observation names to their computed values.
        """
        for reg_key, compute_function in self._register_to_memory.items():
            val = compute_function(self._smac_instance)
            self._memory[reg_key].append(val)
        return {
            obs.name: np.atleast_1d(
                obs.compute(self._smac_instance, self._memory)
            ).astype(np.float32)
            for obs in self._observation_types
        }

    def reset(self) -> None:
        """Reset any stateful observations. Should be called when the env is reset."""
        for obs in self._observation_types:
            if isinstance(obs.compute, GetAFandAcqValue):
                obs.compute.reset()
