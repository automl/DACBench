"""Handle time series features on the observed costs."""

from __future__ import annotations

import numpy as np


def calc_variability(costs: list[float]) -> float:
    """Calculate the variability of the costs.

    Taken from Tierney et al. (2025) (under review).

    Parameters
    ----------
    costs: list[float]
        List of costs.

    Returns:
    -------
    float
        The variability of the costs.
    """
    mean = np.mean(costs)
    g = len(costs)
    summation = [np.mean(costs[:i]) for i in range(1, g - 2)]
    mean_summation = np.mean(summation)
    return mean / mean_summation
