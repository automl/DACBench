"""ParEGO implementation."""

from __future__ import annotations

import numpy as np


class ParEGO:
    """ParEGO implementation based on https://www.cs.bham.ac.uk/~jdk/UKCI-2015.pdf.

    Parameters
    ----------
    n_objectives : int
        Number of objectives to consider
    seed : int
        Random seed
    rho : float , defaults to 0.05
        A small positive value.
    """

    def __init__(self, n_objectives: int, seed: int, rho: float = 0.05) -> None:
        """Initialization of ParEGO.

        Parameters
        ----------
        n_objectives : int
            Number of objectives to consider
        seed : int
            Random seed
        rho : float , defaults to 0.05
            A small positive value.
        """
        self._seed = seed
        self._n_objectives = n_objectives
        self._rho = rho
        self._theta: np.ndarray | None = None
        self._rng = np.random.RandomState(self._seed)

    def _update(self) -> None:
        """Updates the internal state of the ParEGO instance."""
        self._theta = self._rng.rand(self._n_objectives)
        self._theta = self._theta / (np.sum(self._theta) + 1e-10)

    def __call__(self, values: list[float]) -> float:
        """Compute the ParEGO value.

        Parameters
        ----------
        values : list[float]
            Input values.

        Returns:
        -------
        float
            The ParEGO value.
        """
        self._update()
        theta_f = self._theta * np.array(values)
        return float(np.max(theta_f, axis=0) + self._rho * np.sum(theta_f, axis=0))
