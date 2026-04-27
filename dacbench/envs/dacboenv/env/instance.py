"""Instance Selection for DACBO Env."""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class InstanceSelector(ABC):
    """Instance Selector.

    One instance is represented as (task_id, seed).
    The list of instances is [(seed_0, task_id_0), (seed_0, task_id_1), ..., (seed_1, task_id_0),]

    Attributes:
    ----------
    task_ids : list[str]
        List of task ids.
    seeds : list[int]
        List of seeds.
    idx : int
        Current instance index. Default is 0.
    rng : Generator
        Random generator.
    """

    def __init__(
        self, task_ids: list[str], seeds: list[int], selector_seed: int | None = None
    ) -> None:
        """Initialize instance selector.

        Parameters
        ----------
        task_ids : list[str]
            List of task ids.
        seeds : list[int]
            List of seeds.
        selector_seed : int | None, optional
            Selector seed, e.g., needed in random selection, by default None
        """
        self.task_ids = task_ids
        self.seeds = seeds
        self.instances = list(itertools.product(self.seeds, self.task_ids))
        self.idx: int = 0
        self.selector_seed = selector_seed
        self.rng = np.random.default_rng(seed=selector_seed)

    @abstractmethod
    def select_instance(self, size: int = 1) -> tuple[int, str] | list[tuple[int, str]]:
        """Select next instance.

        Parameters
        ----------
        size : int, optional
            The number of instances, by default 1.

        Returns:
        -------
        tuple[int, str] | list[tuple[int, str]]
            (seed, task_id)
        """


class RoundRobinInstanceSelector(InstanceSelector):
    """Round robin instance selector.

    Rotate through instances.
    """

    def __init__(
        self,
        task_ids: list[str],
        seeds: list[int],
        offset: int = 0,
        selector_seed: int | None = None,
    ) -> None:
        """Initialize instance selector.

        Parameters
        ----------
        task_ids : list[str]
            List of task ids.
        seeds : list[int]
            List of seeds.
        offset : int, 0
            An optional offset to add to the index.
        selector_seed : int | None, optional
            Selector seed, e.g., needed in random selection, by default None
        """
        super().__init__(task_ids, seeds, selector_seed)
        self._offset = offset
        self.idx = (self.idx + self._offset) % len(self.instances)

    def select_instance(self, size: int = 1) -> tuple[int, str] | list[tuple[int, str]]:
        """Select next instance.

        Parameters
        ----------
        size : int, optional
            The number of instances, by default 1.

        Returns:
        -------
        tuple[int, str] | list[tuple[int, str]]
            (seed, task_id)
        """
        n_instances = len(self.instances)
        if size == 1:
            instance = self.instances[self.idx]
        else:
            indexer = np.arange(self.idx, self.idx + size) % n_instances
            instance = self.instances[indexer]
        self.idx = (self.idx + size) % n_instances
        return instance


class RandomInstanceSelector(InstanceSelector):
    """Random instance selector."""

    def select_instance(self, size: int = 1) -> tuple[int, str] | list[tuple[int, str]]:
        """Select next instance.

        Parameters
        ----------
        size : int, optional
            The number of instances, by default 1.

        Returns:
        -------
        tuple[int, str] | list[tuple[int, str]]
            (seed, task_id)
        """
        indices = np.arange(0, len(self.instances))
        if size == 1:
            idx = self.rng.choice(indices)
            return self.instances[idx]
        ids = self.rng.choice(indices, size=size)
        return self.instances[ids]


class ExternalInstanceSelector(InstanceSelector):
    """Instance selector for externally managed instance selection (e.g. DACBench).

    Call :meth:`set_instance` before each reset to control which instance
    :meth:`select_instance` returns. Falls back to ``instances[0]`` if no
    instance has been set.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._instance: tuple[int, str] | None = None

    def set_instance(self, instance: tuple[int, str]) -> None:
        """Set the instance returned by the next :meth:`select_instance` call."""
        self._instance = instance

    def select_instance(self, size: int = 1) -> tuple[int, str] | list[tuple[int, str]]:
        """Return the externally set instance without advancing state."""
        instance = (
            self._instance if self._instance is not None else self.instances[self.idx]
        )
        if size == 1:
            return instance
        return [instance] * size
