"""Util functions for handling the Upper Bound Regret (UBR) [Makarova et al., 2022].

The UBR is defined by the estimated worst-case function value of incumbent
minus the estimated lowest function value across search space.
Originally used as a stopping criterion if the difference falls under
a certain thresold. Here we check whether the optimization process
converges.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from ConfigSpace import Configuration
from smac import BlackBoxFacade
from smac.acquisition.maximizer import LocalAndSortedRandomSearch
from smac.model.gaussian_process import GaussianProcess
from smac.model.random_forest import RandomForest
from smac.scenario import Scenario
from smac.utils.logging import get_logger

from dacbench.envs.dacboenv.utils.confidence_bound import LCB, UCB

if TYPE_CHECKING:
    from ConfigSpace import Configuration, ConfigurationSpace
    from smac.main.smbo import SMBO
    from smac.model import AbstractModel
    from smac.runhistory import TrialInfo, TrialValue
from collections.abc import Iterable

logger = get_logger(__name__)


def model_fitted(model: AbstractModel | None) -> bool:
    """Check whether the surrogate model is fitted.

    Parameters
    ----------
    model : AbstractModel
        Surrogate model.

    Returns:
    -------
    bool
        Model fitted or not.
    """
    fitted = False
    if model is not None:
        fitted = (isinstance(model, GaussianProcess) and model._is_trained) or (
            isinstance(model, RandomForest) and model._rf is not None
        )
    return fitted


def calculate_ubr(
    trial_infos: list[TrialInfo] | None,
    trial_values: list[TrialValue] | None,
    configspace: ConfigurationSpace | None,
    seed: int | None = None,
    top_p: float = 0.5,
    smbo: SMBO | None = None,
    model: AbstractModel | None = None,
) -> dict[str, Any]:
    """Calculate the Upper Bound Regret (UBR) from a SMAC optimizer state.

    The UBR is defined by the estimated worst-case function value of incumbent
    minus the estimated lowest function value across search space.
    Originally used as a stopping criterion if the difference falls under
    a certain thresold. Here we check whether the optimization process
    converges.

    Parameters
    ----------
    trial_infos : list[TrialInfo] | None
        Trial information objects corresponding to previously evaluated configurations.
    trial_values : list[TrialValue] | None
        Trial results corresponding to ``trial_infos``.
    configspace : ConfigurationSpace | None
        The search space over which to optimize.
    seed : int | None, optional
        Random seed for reproducibility, by default None.
    top_p : float, optional
        Top p portion of the evaluated configs to be considered by UBR, by default 0.5.
    smbo : SMBO | None, optional
        An existing SMBO instance. If None, a new BlackBoxFacade is initialized with
        the given trials, by default None.
    model : AbstractModel | None
        A model can be passed, for example if the UBR should be calculated on another model
        than from the current SMAC instance.

    Returns:
    -------
    dict[str, Any]
        A dictionary with the following keys:
        - ``"n_evaluated"``: number of evaluated configurations.
        - ``"ubr"``: the computed UBR.
        - ``"min_ucb"``: negative maximum UCB over the selected evaluated configs.
        - ``"min_lcb"``: negative maximum LCB over the config space.
    """

    def dummy_fn(config: Configuration, seed: int | None) -> float:
        return 0

    if smbo is None:
        assert isinstance(trial_infos, Iterable)
        assert isinstance(trial_values, Iterable)

        scenario = Scenario(n_trials=10000, configspace=configspace, seed=seed)
        optimizer = BlackBoxFacade(
            scenario=scenario,
            target_function=dummy_fn,
            overwrite=True,
            logging_level=logging.WARNING,
        )

        # Tell configs
        for info, value in zip(trial_infos, trial_values, strict=True):
            optimizer.tell(info, value)

        smbo = optimizer.optimizer

    model = model or smbo.intensifier.config_selector._model

    # Fit model if still model-free
    # if len(smbo.intensifier.config_selector._initial_design_configs) > 0:
    # if not model_fitted(model):
    #     X, Y, X_configurations = smbo.intensifier.config_selector._collect_data()
    #     smbo.intensifier.config_selector._runhistory.get_configs()
    #     smbo.intensifier.config_selector._model.train(X, Y)
    #     model = smbo.intensifier.config_selector._model

    if model_fitted(model):
        rh = smbo.runhistory
        evaluated_configs = rh.get_configs(sort_by="cost")[:-1]
        evaluated_configs = evaluated_configs[
            : int(np.ceil(len(evaluated_configs) * top_p))
        ]

        ucb_aq = UCB()
        lcb_aq = LCB()

        kwargs = {"model": model, "num_data": rh.finished - 1}
        ucb_aq.update(**kwargs)  # type: ignore[arg-type]
        lcb_aq.update(**kwargs)  # type: ignore[arg-type]

        # Minimize UCB (max -UCB) for all evaluated configs
        acq_values = ucb_aq(evaluated_configs)
        min_ucb = -float(np.squeeze(np.amax(acq_values)))

        # Minimize LCB (max -LCB) on config space
        acq_maximizer = LocalAndSortedRandomSearch(
            configspace=smbo._configspace,
            seed=smbo._scenario.seed,
            acquisition_function=lcb_aq,
        )
        challengers = np.array(
            acq_maximizer._maximize(
                previous_configs=[],
                n_points=1,
            ),
            dtype=object,
        )
        acq_values = challengers[:, 0]
        min_lcb = -float(np.squeeze(np.amax(acq_values)))

        ubr = min_ucb - min_lcb

        return {
            "n_evaluated": smbo.runhistory.finished,
            "ubr": ubr,
            "min_ucb": min_ucb,
            "min_lcb": min_lcb,
        }
    return {}
