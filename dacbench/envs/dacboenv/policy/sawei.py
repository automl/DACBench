"""Moving IQM smoothing for UBR signal."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import trim_mean


def apply_moving_iqm(U: np.ndarray | list, window_size: int = 5) -> np.ndarray:
    """Moving IQM for UBR.

    Smoothes the noisy UBR signal.

    Parameters
    ----------
    U : np.ndarray | list
        UBR history.
    window_size : int, optional
        The window size for smoothing, by default 5.

    Returns:
    -------
    np.ndarray
        Smoothed UBR.
    """

    def moving_iqm(X: np.ndarray) -> float:
        """Apply the IQM to one slice (X) of the UBR.

        Parameters
        ----------
        X : np.ndarray
            One slice of the UBR.

        Returns:
        -------
        float
            IQM of this slice.
        """
        return trim_mean(X, 0.25)

    # Pad UBR so we can apply the sliding window
    U_padded = np.concatenate((np.array([U[0]] * (window_size - 1)), U))
    # Create slices to apply our smoothing method
    slices = sliding_window_view(U_padded, window_size, axis=0)
    # Apply smoothing
    return np.array([moving_iqm(s) for s in slices])
