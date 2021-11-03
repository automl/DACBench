import json
from numbers import Number

from typing import Dict, Union, List, Tuple

import Pyro4
import numpy as np


from dacbench.abstract_env import AbstractEnv
from dacbench.container.container_utils import Encoder, Decoder

NumpyTypes = Union[np.ndarray, np.int, np.float, np.random.RandomState]
DefaultJsonable = Union[
    bool, None, Dict[str, 'DefaultJsonable'], List['DefaultJsonable'], Tuple['DefaultJsonable'], str, float, int]
Jsonable = Union[List['Jsonable'], Dict[str, 'Jsonable'], Tuple['Jsonable'], DefaultJsonable, NumpyTypes]


def json_encode(obj: Jsonable) -> str:
    return json.dumps(obj, indent=None, cls=Encoder)


def json_decode(json_str: str) -> Jsonable:
    return json.loads(json_str, cls=Decoder)


@Pyro4.expose
class RemoteEnvironmentServer:

    def __init__(self, env):
        self.__env: AbstractEnv = env

    def step(self, action: Union[Dict[str, List[Number]], List[Number]]):
        action = json_decode(action)
        json_str = json_encode(self.env.step(action))
        return json_str

    def reset(self):
        state = self.env.reset()
        state = json_encode(state)
        return state

    def render(self, mode="human"):
        # ever used?
        pass

    def close(self):
        self.env.close()

    @property
    def action_space(self):
        return json_encode(self.__env.action_space)



class RemoteEnvironmentClient:

    def __init__(self, env: RemoteEnvironmentServer):
        self.__env = env

    def step(self, action: Union[Dict[str, np.ndarray], np.ndarray]) \
            -> Tuple[Union[Dict[str, np.ndarray], np.ndarray], Number, bool, dict]:
        action = json_encode(action)

        json_str = self.__env.step(action)

        state, reward, done, info = json_decode(json_str)

        return state, reward, done, info

    def reset(self) -> Union[Dict[str, np.ndarray], np.ndarray]:
        state = self.__env.reset()
        state = json_decode(state)
        return state

    def close(self):
        self.__env.close()

    @property
    def action_space(self):
        return json_decode(self.__env.action_space)
