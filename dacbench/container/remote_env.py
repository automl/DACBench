import json
from numbers import Number
from typing import Dict, List, Tuple, Union

import numpy as np
import Pyro4

from dacbench.abstract_env import AbstractEnv
from dacbench.container.container_utils import Decoder, Encoder

NumpyTypes = Union[np.ndarray, np.int32, np.float32, np.random.RandomState]
DefaultJsonable = Union[
    bool,
    None,
    Dict[str, "DefaultJsonable"],
    List["DefaultJsonable"],
    Tuple["DefaultJsonable"],
    str,
    float,
    int,
]
Jsonable = Union[
    List["Jsonable"],
    Dict[str, "Jsonable"],
    Tuple["Jsonable"],
    DefaultJsonable,
    NumpyTypes,
]


def json_encode(obj: Jsonable) -> str:
    """Encode object"""
    return json.dumps(obj, indent=None, cls=Encoder)


def json_decode(json_str: str) -> Jsonable:
    """Decode object"""
    return json.loads(json_str, cls=Decoder)


@Pyro4.expose
class RemoteEnvironmentServer:
    """Server for remote environment."""

    def __init__(self, env):
        """Make env server."""
        self.__env: AbstractEnv = env

    def step(self, action: Union[Dict[str, List[Number]], List[Number]]):
        """Env step."""
        action = json_decode(action)
        json_str = json_encode(self.__env.step(action))
        return json_str

    def reset(self):
        """Reset env."""
        state = self.__env.reset()
        state = json_encode(state)
        return state

    def render(self, mode="human"):
        """Render, does nothing."""
        # ever used?
        pass

    def close(self):
        """Close."""
        self.__env.close()

    @property
    def action_space(self):
        """Return action space."""
        return json_encode(self.__env.action_space)


class RemoteEnvironmentClient:
    """Client for remote environment."""

    def __init__(self, env: RemoteEnvironmentServer):
        """Make client."""
        self.__env = env

    def step(
        self, action: Union[Dict[str, np.ndarray], np.ndarray]
    ) -> Tuple[Union[Dict[str, np.ndarray], np.ndarray], Number, bool, dict]:
        """Remote step."""
        action = json_encode(action)

        json_str = self.__env.step(action)

        state, reward, done, info = json_decode(json_str)

        return state, reward, done, info

    def reset(self) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """Remote reset."""
        state = self.__env.reset()
        state = json_decode(state)
        return state

    def close(self):
        """Close."""
        self.__env.close()

    @property
    def action_space(self):
        """Return env action space."""
        return json_decode(self.__env.action_space)
