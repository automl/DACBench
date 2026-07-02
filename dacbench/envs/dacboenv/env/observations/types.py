"""Data Types for Observations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from gymnasium.spaces import Space
    from smac.main.smbo import SMBO


Memory = dict[str, list[float]]

ObsType = dict[str, Any]


@dataclass
class ObservationType:
    """Represents a single observation type.

    Attributes:
    ----------
    name : str
        Name of the observation.
    space : Space
        Gymnasium space for the observation's value range and type.
    compute : Callable[[SMBO], Any]
        Function to compute the observation value from a SMAC instance.
    default : int | float
        The observation's default value.
    """

    name: str
    space: Space
    compute: Callable[[SMBO, Memory | None], Any]
    default: Any


@dataclass
class MultiObservationType:
    """Represents a multi observation type.
    A multi observation is a collection of observation types that are created together.

    Attributes:
    ----------
    name : str
        Name of the observation.
    create : Callable[[SMBO], Sequence[ObservationType]]
        Function to create the collection of ObservervationTypes from a SMAC instance.
    """

    name: str
    create: Callable[[SMBO], Sequence[ObservationType]]
