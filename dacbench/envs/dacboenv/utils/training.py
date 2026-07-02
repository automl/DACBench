"""SMAC surrogate model training helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smac.intensifier.config_selector import ConfigSelector
    from smac.model import AbstractModel


def ensure_model_trained(
    config_selector: ConfigSelector, model: AbstractModel | None
) -> bool:
    """Trigger training of the surrogate model if it hasn't been fitted yet.

    In SMAC 3, ``tell()`` updates the RunHistory but defers model training until
    the next ``ask()``. This helper ensures that data collected via ``tell()``
    is immediately reflected in the surrogate model, which is critical because
    DACBO observations (e.g. acquisition values, UBR) are queried immediately
    after ``tell()`` before the next ``ask()`` occurs.

    Parameters
    ----------
    config_selector
        The SMAC config selector instance (provides ``_collect_data`` and ``_model``).
    model
        The current surrogate model, or ``None`` (falls back to ``config_selector._model``).

    Returns:
    -------
    bool
        ``True`` if the model was newly trained, ``False`` otherwise.
    """
    current_model = model or config_selector._model
    if (
        current_model is None
        or not hasattr(current_model, "train")
        or current_model._is_trained
    ):
        return False

    X, Y, _ = config_selector._collect_data()
    if X.shape[0] > 0:
        current_model.train(X, Y)
        return True
    return False
