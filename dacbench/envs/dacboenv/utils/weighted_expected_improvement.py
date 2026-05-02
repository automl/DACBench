"""Handling the Weighted Expected Improvement Acquisition Function (WEI) [Sobester et al., 2005]."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm
from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.utils.logging import get_logger

logger = get_logger(__name__)


class WEI(AbstractAcquisitionFunction):
    r"""Weighted Expected Improvement (WEI) acquisition function.

    WEI [Sobester et al., 2005] is Expected Improvement (EI) [Mockus et al., 1978] but its
    two terms are weighted by alpha. One term is more exploratory, the other more
    exploitative.
    alpha = 0.5 recovers the standard EI [Mockus et al., 1978]
    alpha = 1 has similar behavior as $PI(x) = \\Phi( z(x))$ [Kushner, 1974]
    alpha = 0 emphasizes a stronger exploration

    Attributes:
    ----------
    _xi : float
        Exploration-exploitation trade-off parameter.
    _log : bool
        Whether the function operates in log-space. Not implemented yet.
    _eta : float | None
        Current best function value, set during ``update``.
    _alpha : float
        Weighting parameter that interpolates between PI and EI.
    _use_pure_PI : bool
        Whether to enforce pure PI mode.
    pi_term : np.ndarray | None
        Computed PI component values.
    pi_pure_term : np.ndarray | None
        Pure PI component values.
    pi_mod_term : np.ndarray | None
        Modified PI component values ``(eta - mu - xi) * Phi(z)``.
    ei_term : np.ndarray | None
        Computed EI component values.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        xi: float = 0,
        log: bool = False,
        use_pure_PI: bool = False,
    ) -> None:
        """Initialize the WEI acquisition function.

        Parameters
        ----------
        alpha : float, optional
            The initial weight of weighted expected improvement, by default 0.5.
            This equals EI.
        xi : float, optional
            Exploration-exploitation trade-off parameter. Default is 0.
        log : bool, optional
            Whether to operate in log-space. Not implemented. Default is False.
        use_pure_PI : bool, optional
            If True, enforces pure PI behavior. Default is False.
        """
        super().__init__()
        self._xi: float = xi
        self._log: bool = log
        if self._log:
            raise NotImplementedError
        self._eta: float | None = None
        self._alpha = alpha
        self._use_pure_PI = use_pure_PI

        self.pi_term: np.ndarray | None = None
        self.pi_pure_term: np.ndarray | None = None
        self.pi_mod_term: np.ndarray | None = None
        self.ei_term: np.ndarray | None = None

    @property
    def name(self) -> str:  # noqa: D102
        return "Weighted Expected Improvement"

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "xi": self._xi,
                "log": self._log,
                "alpha": self._alpha,
            }
        )

        return meta

    def _update(self, **kwargs: Any) -> None:
        """Update acsquisition function attributes.

        Parameters
        ----------
        eta : float
            Function value of current incumbent.
        xi : float, optional
            Exploration-exploitation trade-off parameter
        """
        assert "eta" in kwargs
        self._eta = kwargs["eta"]

        if "xi" in kwargs and kwargs["xi"] is not None:
            self._xi = kwargs["xi"]
        alpha = kwargs.get("alpha")
        if alpha is not None:
            self._alpha = alpha

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Compute EI acquisition value.

        Parameters
        ----------
        X : np.ndarray [N, D]
            The input points where the acquisition function should be evaluated. The dimensionality of X is (N, D),
            with N as the number of points to evaluate at and D is the number of dimensions of one X.

        Returns:
        -------
        np.ndarray [N,1]
            Acquisition function values wrt X.

        Raises:
        ------
        ValueError
            If `update` has not been called before (current incumbent value `eta` unspecified).
        ValueError
            If EI is < 0 for at least one sample (normal function value space).
        ValueError
            If EI is < 0 for at least one sample (log function value space).
        """
        assert self._model is not None
        assert self._xi is not None
        if self._use_pure_PI:
            assert self._alpha == 1.0, (
                f"{self._alpha} != 1.0 with pure PI. Any other combination leads to wrong behavior."
            )

        if self._eta is None:
            raise ValueError(
                "No current best specified. Call update("
                "eta=<int>) to inform the acquisition function "
                "about the current best value."
            )

        if not self._log:
            if len(X.shape) == 1:
                X = X[:, np.newaxis]

            m, v = self._model.predict_marginalized(X)
            s = np.sqrt(v)

            def calculate_f() -> np.ndarray:
                z = (self._eta - m - self._xi) / s
                pi_term = (
                    norm.cdf(z)
                    if self._use_pure_PI
                    else (self._eta - m - self._xi) * norm.cdf(z)
                )
                ei_term = s * norm.pdf(z)
                self.pi_term = pi_term
                self.pi_pure_term = norm.cdf(z)
                self.pi_mod_term = (self._eta - m - self._xi) * norm.cdf(z)
                self.ei_term = ei_term
                return self._alpha * pi_term + (1 - self._alpha) * ei_term

            if np.any(s == 0.0):
                # if std is zero, we have observed x on all instances
                # using a RF, std should be never exactly 0.0
                # Avoid zero division by setting all zeros in s to one.
                # Consider the corresponding results in f to be zero.
                logger.warning("Predicted std is 0.0 for at least one sample.")
                s_copy = np.copy(s)
                s[s_copy == 0.0] = 1.0
                f = calculate_f()
                f[s_copy == 0.0] = 0.0
            else:
                f = calculate_f()

            # if (f < 0).any():
            #     # TODO is it okay if this acq fun is smaller than 0?
            #     logger.warn("Expected Improvement is smaller than 0 for at least one " "sample.")
            #     # raise ValueError("Expected Improvement is smaller than 0 for at least one " "sample.")
            return f
        raise NotImplementedError
