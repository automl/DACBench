"""Action utilities for DACBOEnv."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Space
from smac.acquisition.function import EI, PI
from smac.main.config_selector import ConfigSelector
from smac.main.smbo import SMBO

from dacbench.envs.dacboenv.utils.confidence_bound import UCB
from dacbench.envs.dacboenv.utils.weighted_expected_improvement import WEI

if TYPE_CHECKING:
    from smac.acquisition.function.abstract_acquisition_function import (
        AbstractAcquisitionFunction,
    )
    from smac.main.smbo import SMBO

    from dacbench.envs.dacboenv.dacboenv import ActType


@dataclass
class ParameterAction:
    """Represents a parameter action for a fixed acquisition function.

    Attributes:
    ----------
    attr : str
        Name of the function object's attribute.
    space : Space
        Gymnasium space for the parameter's value range and type.
    log : bool, optional
        Whether the parameter is interpreted in log scale.
    name : str
        String representation of the action.
    """

    attr: str
    space: Space
    log: bool = False
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"ParameterAction:{self.attr}"


@dataclass
class FunctionAction:
    """Represents an action for selecting an acquisition function.

    Attributes:
    ----------
    space : Space
        Gymnasium space for the discrete selection of acquisition functions.
    name : str
        String representation of the action.
    """

    space: Space
    name: str = field(init=False, default="FunctionAction")


ActionType = ParameterAction | FunctionAction


class AbstractActionSpace:
    """Manages action spaces the DACBOenv.

    Parameters
    ----------
    smac_instance : SMBO
        The SMAC instance.

    Attributes:
    ----------
    _smac_instance : SMBO
        Reference to the associated SMAC instance.
    _action : ActionType
        The action object defining the action space.
    _action_space : Space
        The Gymnasium space for the current action configuration.
    """

    def __init__(self, smac_instance: SMBO) -> None:
        """Initialize the ActionSpace.

        Parameters
        ----------
        smac_instance : SMBO
            The SMAC instance.

        """
        self._smac_instance = smac_instance
        self._action = self._create_action()
        self._action_space = self._action.space

    @abstractmethod
    def _create_action(self) -> ActionType:
        """Create the appropriate action object.

        Returns:
        -------
        ActionType
            The action object.
        """
        raise NotImplementedError

    @abstractmethod
    def update_optimizer(self, action: ActType) -> None:
        """Update the SMAC optimizer based on the chosen action.

        Parameters
        ----------
        action : ActType
            The action according to a policy.
        """
        raise NotImplementedError

    @property
    def space(self) -> Space:
        """Returns the Gymnasium space for the action.

        Returns:
        -------
        Space
            The action space.
        """
        return self._action_space


class WEITempoRLActionSpace(AbstractActionSpace):
    """TempoRL Action Space for WEI.

    The first action is the skip duration, the second the action to hold.
    This might prevent wild oscillating actions/parameter values.
    """

    def __init__(
        self,
        smac_instance: SMBO,
        step_durations: list[int] | None,
        param_levels: list[float] | None = None,
    ) -> None:
        self._step_durations = (
            list(step_durations) if step_durations is not None else [1, 5, 10]
        )
        if any(d <= 0 for d in self._step_durations):
            raise ValueError(
                f"All step_durations must be > 0, got {self._step_durations}"
            )
        self._param_levels = (
            list(param_levels)
            if param_levels is not None
            else [0.0, 0.25, 0.5, 0.75, 1]
        )
        super().__init__(smac_instance)

    def _create_action(self) -> ParameterAction | FunctionAction:
        nvec = [len(self._step_durations), len(self._param_levels)]
        return ParameterAction(attr="_alpha", space=MultiDiscrete(nvec=nvec), log=False)

    def update_optimizer(self, action: ActType) -> None:
        """Update the acquisition function parameter value.

        Parameters
        ----------
        action : ActType
            A single numeric action value for the parameter.
        """
        assert isinstance(action, Sequence)
        assert isinstance(
            self._smac_instance._intensifier._config_selector, ConfigSelector
        )

        param_level_idx = int(action[1])
        param_val = self._param_levels[param_level_idx]

        setattr(
            self._smac_instance._intensifier._config_selector._acquisition_function,
            self._action.attr,  # type: ignore[union-attr]
            param_val,
        )


class AcqParameterActionSpace(AbstractActionSpace):
    """Action space for tuning parameters of the current acquisition function.

    Attributes:
    ----------
    _PARAMETERS : ClassVar[dict[type[AbstractAcquisitionFunction], ParameterAction]]
        Mapping of acquisition function classes to their corresponding parameter actions.
    """

    _ATTRIBUTE_MAP: ClassVar[dict[type[AbstractAcquisitionFunction], str]] = {
        EI: "_xi",
        PI: "_xi",
        UCB: "_beta",
        WEI: "_alpha",
    }
    _LOG: ClassVar[dict[type[AbstractAcquisitionFunction], bool]] = {
        EI: False,
        PI: False,
        UCB: True,
        WEI: False,
    }

    def __init__(
        self,
        smac_instance: SMBO,
        bounds: tuple[int, int] | tuple[float, float],
        adjustment_type: str = "continuous",
        step_size: float = 0.5,
    ) -> None:
        """Initialize action space.

        Parameters
        ----------
        smac_instance : SMBO
            The smac instance.
        bounds : tuple[int, int] | tuple[float, float]
            The action space bounds (low, high). If the acquisition function hyperparameter should be adjusted in log
            space, it is assumed that the bounds already are in log space.
            For EI and PI, usually the bounds are (-10, 10). For UCB: -6 to 3 in log10 space
            (for continuous and bucket).
        adjustment_type : str, optional
            The adjustment, by default "continuous". Can be continuous, bucket or step.
            For bucket, we have discrete choices with bounds as bounds.
            For step, the lower bound is interpreted as the
            decrease (but put a negative number as everything is just added), the upper as increase, and there will be
            a do nothing action.
        step_size : float, optional
            If the adjustment type is step, we have as actions: decrease, do nothing, increase. For the amount of
            decrease/increase we need to specify the step size.
        """
        self._last: float = 0.0
        self._adjustment_type = adjustment_type
        self._bounds = bounds
        self._step_size = step_size
        super().__init__(smac_instance)

    def _create_action(self) -> ParameterAction:
        """Create a ParameterAction for the current acquisition function.

        Returns:
        -------
        ParameterAction
            The parameter action object for the selected acquisition function.

        Raises:
        ------
        ValueError
            If the acquisition function of the SMAC instance is unsupported.
        """
        acquisition_function = (
            self._smac_instance._intensifier._config_selector._acquisition_function
        )
        if isinstance(acquisition_function, UCB) and acquisition_function._update_beta:
            raise ValueError(
                "For UCB we can only adjust beta and for this, `_update_beta` must be set to False."
                "If you mean to adjust nu, please add this in the code."
            )

        attribute = self._ATTRIBUTE_MAP[type(acquisition_function)]
        is_log = self._LOG[type(acquisition_function)]

        if self._adjustment_type in {"continuous", "continuousstep"}:
            dacbo_action_space = ParameterAction(
                attr=attribute,
                space=Box(low=self._bounds[0], high=self._bounds[1], dtype=np.float32),
                log=is_log,
            )
        elif self._adjustment_type == "step":
            dacbo_action_space = ParameterAction(
                attr=attribute, space=Discrete(n=3), log=is_log
            )
        elif self._adjustment_type == "bucket":
            if not isinstance(self._bounds[0], int):
                raise ValueError(
                    "Expected self._bounds[0] to be int for 'bucket' adjustment type, "
                    f"got {type(self._bounds[0]).__name__}"
                )
            if not isinstance(self._bounds[1], int):
                raise ValueError(
                    "Expected self._bounds[1] to be int for 'bucket' adjustment type, "
                    f"got {type(self._bounds[1]).__name__}"
                )
            n = abs(self._bounds[0]) + self._bounds[1] + 1
            dacbo_action_space = ParameterAction(
                attr=attribute, space=Discrete(n=n), log=is_log
            )
        else:
            raise ValueError(f"Unknown adjustment type: {self._adjustment_type}.")

        return dacbo_action_space

    def update_optimizer(self, action: ActType) -> None:
        """Update the acquisition function parameter value.

        Parameters
        ----------
        action : ActType
            A single numeric action value for the parameter.
        """
        action_val = np.array(action).item()

        if self._adjustment_type == "continuous":
            param_val = action_val
        elif self._adjustment_type == "continuousstep":
            self._last = np.clip(
                self._last + action_val, self._bounds[0], self._bounds[1]
            )
            param_val = self._last
        elif self._adjustment_type == "step":
            if action_val == 0:
                self._last -= self._step_size
            elif action_val == 1:
                self._last = self._last
            elif action_val == 2:
                self._last += self._step_size

            self._last = np.clip(self._last, self._bounds[0], self._bounds[1])
            param_val = self._last
        elif self._adjustment_type == "bucket":
            param_val = (
                action_val + self._bounds[0]
            )  # that value probably is below 0 so basically the offset

        if self._action.log:  # type: ignore[union-attr]
            param_val = 10**param_val

        setattr(
            self._smac_instance._intensifier._config_selector._acquisition_function,
            self._action.attr,  # type: ignore[union-attr]
            param_val,
        )


class AcqFunctionActionSpace(AbstractActionSpace):
    """Action space for selecting an acquisition function.

    Attributes:
    ----------
    _acq_fun_dict : ClassVar[dict[int, type[AbstractAcquisitionFunction]]]
        Mapping of integer IDs to available acquisition function classes.
    """

    def __init__(
        self,
        smac_instance: SMBO,
        acquisition_functions: list[AbstractAcquisitionFunction] | None = None,
    ) -> None:
        """Initialize discrete acquisition function choice space.

        Parameters
        ----------
        smac_instance : SMBO
            The smac instance.
        acquisition_functions : list[AbstractAcquisitionFunction] | None, optional
            List of acquisition function classes, by default None. If None, will be [EI, PI, UCB].
        """
        _afs = [EI, PI, UCB] if acquisition_functions is None else acquisition_functions
        self._acq_fun_dict = dict(enumerate(_afs))
        super().__init__(smac_instance)

    def _create_action(self) -> FunctionAction:
        """Create a FunctionAction representing the discrete selection of acquisition functions.

        Returns:
        -------
        FunctionAction
            The FunctionAction object for acquisition function selection.
        """
        return FunctionAction(Discrete(len(self._acq_fun_dict)))

    def update_optimizer(self, action: ActType) -> None:
        """Update the SMAC optimizer to use the selected acquisition function.

        Parameters
        ----------
        action : ActType
            Integer index representing the selected acquisition function.
        """
        function_idx = int(np.array(action).item())
        self._smac_instance.update_acquisition_function(
            acquisition_function=self._acq_fun_dict[function_idx]()
        )
