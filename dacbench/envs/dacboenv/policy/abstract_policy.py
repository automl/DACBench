"""Abstract Policy."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, TypeAlias

    from dacbench.envs.dacboenv.dacboenv import ActType, DACBOEnv
    from dacbench.envs.dacboenv.env.observations.types import ObsType


class AbstractPolicy:
    """Abstract base class for DACBOEnv policies.

    A policy defines a mapping from observations to actions within
    the DACBO environment.
    """

    def __init__(self, env: DACBOEnv, **kwargs: Any) -> None:
        """Initialize the policy.

        Parameters
        ----------
        env : DACBOEnv
            The environment in which the policy operates.
        **kwargs : Any
            Keyword arguments from child classes.
        """
        self._env = env
        self._init_kwargs = kwargs.copy()

    def get_init_kwargs(self) -> dict:
        """Get kwargs from initialization.

        Requirement is that each child class passes their kwargs to super.
        """
        return self._init_kwargs

    @abstractmethod
    def __call__(self, obs: ObsType) -> ActType:
        """Select an action given the current observation.

        Parameters
        ----------
        obs : ObsType
            The current environment observation.

        Returns:
        -------
        ActType
            The selected action.
        """
        raise NotImplementedError

    def set_seed(self, seed: int | None) -> None:
        """Set seed for stochastic policies.

        Parameters
        ----------
        seed : int | None
            Seed
        """


Policy: TypeAlias = AbstractPolicy
