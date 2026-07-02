"""Math helper functions."""

from __future__ import annotations

import numpy as np


def symlog(
    x: np.ndarray, linthresh: float = 1.0, base: float = 10, eps: float = 1e-10
) -> np.ndarray:
    """Apply a symmetric logarithmic (symlog) transformation.

    This transformation behaves linearly for values close to zero and
    logarithmically for values with magnitude larger than `linthresh`.
    It is useful for visualizing or scaling data that spans several
    orders of magnitude while preserving the sign of the input.

    Parameters
    ----------
    x : array_like
        Input value or array of values to transform.
    linthresh : float, optional
        Threshold below which the transformation is linear.
        Must be positive. Default is 1.0.
    base : float, optional
        Logarithmic base used for the transformation. Default is 10.
    eps : float, optional
        Small value to avoid log(0).

    Returns:
    -------
    ndarray
        Transformed values with the same shape as `x`.

    Notes:
    -----
    The transformation is defined as:

        symlog(x) = sign(x) * (|x| / linthresh)              if |x| <= linthresh
                    sign(x) * (1 + log_base(|x| / linthresh)) otherwise

    This function is similar to Matplotlib's ``symlog`` scale.
    """
    x = np.asarray(x)
    sign = np.sign(x)
    abs_x = np.abs(x)

    return sign * np.where(
        abs_x <= linthresh,
        abs_x / linthresh,
        1 + np.log(np.maximum(abs_x / linthresh, eps)) / np.log(base),
    )


def safe_log10(x: np.ndarray | float, eps: float = 1e-10) -> float:
    """Computes a numerically safe logarithm of x.

    Parameters
    ----------
    - x : array-like or scalar
    - eps : float, small value to avoid log10(0)

    Returns:
    -------
    - log10(x) safely
    """
    x = np.asarray(x)
    return np.log10(np.maximum(x, eps))


def sigmoid(z: float | np.ndarray) -> float | np.ndarray:
    """Sigmoid function.

    Parameters
    ----------
    z : scalar or array-like
        Input

    Returns:
    -------
    scalar or array-like
        Sigmoid of input.
    """
    return 1 / (1 + np.exp(-z))
