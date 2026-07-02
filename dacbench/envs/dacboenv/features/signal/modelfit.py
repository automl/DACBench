"""Assessing a model's global fit."""

from __future__ import annotations

import copy
import time
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold

from dacbench.envs.dacboenv.utils.loggingutils import get_logger

if TYPE_CHECKING:
    from smac.main.smbo import SMBO

from typing import Any

from smac.facade.hyperparameter_optimization_facade import (
    HyperparameterOptimizationFacade,
)

from dacbench.envs.dacboenv.features.signal.ubr import calculate_ubr

logger = get_logger(__name__)


def gp_nll(mean: np.array, var: np.array, y_test: np.array) -> float:
    """Computes the mean negative log-likelihood (NLL) for Gaussian predictions.

    Assumes the predictions are samples from a Gaussian Process (GP),
    and thus follow a normal distribution with the given means and variances.

    Parameters
    ----------
    mean : np.array
        Predicted mean values of the distribution.
    var : np.array
        Predicted variances corresponding to the mean values.
    y_test : np.array
        Ground truth target values.

    Returns:
    -------
    float
        The mean negative log-likelihood.
    """
    mean = mean.squeeze()
    var = var.squeeze()
    nll = 0.5 * np.log(2 * np.pi * var) + ((y_test - mean) ** 2) / (2 * var)
    return np.mean(nll)


def calculate_model_fit(
    smbo: SMBO,
    k: int = 10,
    top_proportion: float = 1.0,
    metrics: list[str] | None = None,
    model_type: str | None = None,
) -> dict[str, Any]:
    """Computes a model's global fit with regard to given metrics.

    Results are in the same order as the given metrics.

    Parameters
    ----------
    smbo : SMBO
        SMAC instance
    k : int
        How many folds to use for cross-validation
    top_proportion : float
        Only use the top_proportion best-performing configurations for evaluation
    metric : list[str]
        Which metrics to use for assessing the fit
    model_type : str
        Which model to use for assessing the fit
    """
    if metrics is None:
        metrics = ["mse"]
    X, y, _ = smbo.intensifier.config_selector._collect_data()

    # Sort by cost (argsort yields low to high costs)
    sort_indices = np.argsort(y.squeeze())
    X = X[sort_indices].squeeze()
    y = y[sort_indices].squeeze()

    cost_min = float(np.min(y))
    cost_max = float(np.max(y))
    threshold = cost_min + (cost_max - cost_min) * top_proportion

    cost_lower_than_threshold = y < threshold

    X = X[cost_lower_than_threshold]
    y = y[cost_lower_than_threshold]

    # If less than k samples: LOOCV
    k = min(len(X), k)

    if k < 3 or len(X) != len(y):
        return {
            "mean_scores": np.nan,
            "scores": [],
            "total_range": np.nan,
            "cost_min": cost_min,
            "cost_max": cost_max,
            "n_configs": len(X),
            "rel_means": np.nan,
            "duration": 0,
            "k": k,
            "top_proportion": top_proportion,
            "metrics": metrics,
        }

    kf = KFold(n_splits=k, shuffle=True)
    all_scores: list[list[float]] = [[] for _ in metrics]

    config_selector = smbo.intensifier.config_selector

    score_functions = []

    metric_map = {
        "mse": mean_squared_error,
        "rmse": root_mean_squared_error,
        "mae": mean_absolute_error,
        "mape": mean_absolute_percentage_error,
        "r2": r2_score,
        "ubr": calculate_ubr,
        "nll": gp_nll,
    }

    try:
        score_functions = [metric_map[metric] for metric in metrics]
    except KeyError as e:
        raise ValueError(f"Unknown metric: {e.args[0]}") from e

    start = time.time()
    for _i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if model_type is None:
            model = copy.deepcopy(config_selector._model)
        elif model_type == "rf":
            model = HyperparameterOptimizationFacade.get_model(
                config_selector._scenario
            )
        else:
            raise ValueError(f"Unknown model: {model_type}")

        model.train(X_train, y_train)

        y_pred, var = model.predict(X_test)

        for j, score_function in enumerate(score_functions):
            if score_function == calculate_ubr:
                continue

            if score_function == gp_nll:
                score = score_function(y_pred, var, y_test)
            elif score_function != r2_score or (
                score_function == r2_score and len(y_test) > 1
            ):
                score = score_function(y_pred, y_test)
            else:
                score = np.nan  # R2 is ill-defined with less than 2 samples. Oh well
            all_scores[j].append(score)

    end = time.time()
    duration = end - start

    # Handle ubr
    if "ubr" in metrics:
        all_scores[metrics.index("ubr")] = np.append(
            all_scores[metrics.index("ubr")],
            [
                calculate_ubr(
                    trial_infos=None,
                    trial_values=None,
                    configspace=None,
                    seed=None,
                    smbo=smbo,
                )["ubr"]
            ]
            * len(all_scores[0]),
        )

    mean_scores = np.mean(all_scores, axis=1)
    total_range = cost_max - cost_min
    rel_means = mean_scores / total_range

    mean_str = " ".join(f"{x:.4f}" for x in mean_scores)
    rel_str = " ".join(f"{x:.4f}" for x in rel_means)
    logger.debug(
        f"CV ({k} folds): means: [{mean_str}], rel_means: [{rel_str}], time: {duration:.2f}s"
    )

    # How to assess that we have a globally good model?
    return {
        "mean_scores": mean_scores,
        "scores": all_scores,
        "total_range": total_range,
        "cost_min": cost_min,
        "cost_max": cost_max,
        "n_configs": len(X),
        "rel_means": rel_means,
        "duration": duration,
        "k": k,
        "top_proportion": top_proportion,
        "metrics": metrics,
    }
