"""Observations regarding acquisition function values for DACBOEnv."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

import numpy as np
from gymnasium.spaces import Box
from smac.acquisition.function import EI, PI
from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.main.smbo import SMBO

from dacbench.envs.dacboenv.env.observations.types import ObservationType
from dacbench.envs.dacboenv.features.signal.ubr import model_fitted
from dacbench.envs.dacboenv.utils.weighted_expected_improvement import WEI

if TYPE_CHECKING:
    from ConfigSpace import Configuration
    from smac.acquisition.function.abstract_acquisition_function import (
        AbstractAcquisitionFunction,
    )
    from smac.main.smbo import SMBO
    from smac.model import AbstractModel
    from smac.runhistory.runhistory import RunHistory

    from dacbench.envs.dacboenv.env.observations.types import Memory


def get_acq_value(
    solver: SMBO, acq_fun_class: AbstractAcquisitionFunction
) -> float | None:
    """Get the acquisition function value for the last added configuration.

    Parameters
    ----------
    solver : SMBO
        The SMAC solver instance.
    acq_fun_class : AbstractAcquisitionFunction
        The acquisition function class.

    Returns:
    -------
    float | None
        The acquisition value for the last configuration, or None, if the model has not been fitted yet.
    """
    retval = get_af_and_acq_value(solver=solver, acq_fun_class=acq_fun_class)
    if retval is not None:
        _acq_fun, acq_value = retval
        return acq_value
    return None


# TODO Build proper stateful abstract class
class GetAFandAcqValue:
    """Get the AF and query values using the previous incumbent.

    This is important because in ask, first a new model is trained, and then
    a configuration is proposed. We want to have the AF value for this configuration
    on that model.
    Then, the objective function is evaluated, and tell begins.
    In tell, we gather the observations for the dacboenv, after the tell of SMAC.
    In SMAC tell, the incumbent is updated, thus we need to take the incumbent from the
    previous iteration, to use the same AF as used for finding the next configuration.
    The incumbent cost is eta and influences the AF vals.
    """

    def __init__(self) -> None:
        self._model = None
        self._incumbent = None

    def reset(self) -> None:
        """Reset incumbent in case of environment reset."""
        self._incumbent = None

    def __call__(
        self, solver: SMBO, acq_fun_class: AbstractAcquisitionFunction
    ) -> tuple[AbstractAcquisitionFunction, float] | None:
        """Calculate the AF value.

        Parameters
        ----------
        solver : SMBO
            The current SMAC instance.
        acq_fun_class : AbstractAcquisitionFunction
            The acquisiton function class.

        Returns:
        -------
        tuple[AbstractAcquisitionFunction, float] | None
            The evaluated AF, together with the value for the last proposed/evaluated config.
        """
        retval = get_af_and_acq_value(
            solver=solver,
            acq_fun_class=acq_fun_class,
            model=self._model,
            incumbent=self._incumbent,
        )
        # self._model = deepcopy(solver.intensifier.config_selector._model)
        self._incumbent = solver.intensifier.get_incumbents(sort_by="cost")[0]
        return retval


# TODO Each of the classes makes its own deepcopy of the model. This could be optimized further.
class GetAcqValue(GetAFandAcqValue):
    """Get the acq value for an acquisition function."""

    def __call__(
        self, solver: SMBO, acq_fun_class: AbstractAcquisitionFunction
    ) -> float | None:  # type: ignore[override]
        """Get the acquisition function value for the last added configuration.

        Parameters
        ----------
        solver : SMBO
            The SMAC solver instance.
        acq_fun_class : AbstractAcquisitionFunction
            The acquisition function class.

        Returns:
        -------
        float | None
            The acquisition value for the last configuration, or None, if the model has not been fitted yet.
        """
        retval = super().__call__(solver=solver, acq_fun_class=acq_fun_class)
        if retval is not None:
            _acq_fun, acq_value = retval
            return acq_value
        return None


class GetAcqValueEI(GetAcqValue):
    """Get the acq value for EI."""

    def __call__(self, solver: SMBO, memory: Memory | None = None) -> float | None:  # type: ignore[override]
        """Get acquisiton function value for last configuration with EI acquisition function.

        Parameters
        ----------
        solver : SMBO
            The SMAC instance.
        memory : Memory, optional
            Unused memory.

        Returns:
        -------
        float | None
            The acquisition value, or None, if the model has not been fitted yet.
        """
        return super().__call__(solver, EI)


class GetAcqValuePI(GetAcqValue):
    """Get the acq value for PI."""

    def __call__(self, solver: SMBO, memory: Memory | None = None) -> float | None:  # type: ignore[override]
        """Get acquisiton function value for last configuration with PI acquisition function.

        Parameters
        ----------
        solver : SMBO
            The SMAC instance.
        memory : Memory, optional
            Unused memory.

        Returns:
        -------
        float | None
            The acquisition value, or None, if the model has not been fitted yet.
        """
        return super().__call__(solver, PI)


class GetAcqValueWEI(GetAcqValue):
    """Get the acq value for WEI."""

    def __call__(self, solver: SMBO, memory: Memory | None = None) -> float | None:  # type: ignore[override]
        """Get acquisiton function value for last configuration with WEI acquisition function.

        Parameters
        ----------
        solver : SMBO
            The SMAC instance.
        memory : Memory, optional
            Unused memory.

        Returns:
        -------
        float | None
            The acquisition value, or None, if the model has not been fitted yet.
        """
        return super().__call__(solver, WEI)


class GetAcqValueWEIExplore(GetAFandAcqValue):
    """Get the acq value for the exploration term of WEI."""

    def __call__(self, solver: SMBO, memory: Memory | None = None) -> float | None:  # type: ignore[override]
        """Get the exploration term of WEI for last configuration.

        Parameters
        ----------
        solver : SMBO
            The SMAC instance.
        memory : Memory, optional
            Unused memory.

        Returns:
        -------
        float | None
            The exploration term of WEI, or None, if the model has not been fitted yet.
        """
        retval = super().__call__(solver=solver, acq_fun_class=WEI)
        if retval is not None:
            acq_fun, _acq_value = retval
            _pi_term = acq_fun.pi_pure_term[0][0]  # type: ignore[union-attr,index]
            ei_term = acq_fun.ei_term[0][0]  # type: ignore[union-attr,index]
            return ei_term  # noqa: RET504
        return None


def get_af_and_acq_value(
    solver: SMBO,
    acq_fun_class: AbstractAcquisitionFunction,
    model: AbstractModel | None = None,
    incumbent: Configuration | None = None,
) -> tuple[AbstractAcquisitionFunction, float] | None:
    """Get the acquisition function value for the last added configuration.

    Parameters
    ----------
    solver : SMBO
        The SMAC solver instance.
    acq_fun_class : AbstractAcquisitionFunction
        The acquisition function class.

    Returns:
    -------
    tuple[AbstractAcquisitionFunction, float] | None
        The acquisition function, and the acquisition value for the last configuration, or None, if the model has not
        been fitted yet.
    """
    config_selector = solver._intensifier._config_selector
    model = model or config_selector._model
    retval = None
    if model_fitted(model):
        rh: RunHistory = config_selector._runhistory
        incumbent_previous = (
            incumbent or solver.intensifier.get_incumbents(sort_by="cost")[0]
        )
        incumbent_current = solver.intensifier.get_incumbents(sort_by="cost")[0]
        _eta_current = config_selector._runhistory.get_cost(incumbent_current)
        eta_previous = config_selector._runhistory.get_cost(incumbent_previous)
        eta = eta_previous

        acq_fun: AbstractAcquisitionFunction = acq_fun_class()
        acq_fun.update(model=model, eta=eta)
        trial_key = list(rh.keys())[-1]
        config_id = trial_key.config_id
        config = rh.get_config(config_id)
        acq_value = acq_fun([config])[0][0]
        retval = acq_fun, acq_value
    return retval


acq_value_wei_explore_observation = ObservationType(
    name="acq_value_WEI_explore",
    space=Box(low=0, high=np.inf, dtype=np.float32),
    compute=GetAcqValueWEIExplore,  # type: ignore[arg-type]
    default=0,
)

acq_value_ei_observation = ObservationType(
    name="acq_value_EI",
    space=Box(low=0, high=np.inf, dtype=np.float32),
    compute=GetAcqValueEI,  # type: ignore[arg-type]
    default=0,
)

acq_value_wei_observation = ObservationType(
    name="acq_value_WEI",
    space=Box(low=0, high=np.inf, dtype=np.float32),
    compute=GetAcqValueWEI,  # type: ignore[arg-type]
    default=0,
)

acq_value_pi_observation = ObservationType(
    name="acq_value_PI",
    space=Box(low=0, high=1, dtype=np.float32),
    compute=GetAcqValuePI,  # type: ignore[arg-type]
    default=0,
)
