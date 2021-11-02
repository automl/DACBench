from numbers import Number
from typing import Dict, Union, List, Tuple

import numpy as np


from dacbench.abstract_env import AbstractEnv


def from_space_to_simple(
        self, sample: Union[Dict[str, np.ndarray], np.ndarray]
) -> Union[Dict[str, List[Number]], List[Number]]:
    # this way is ok
    raise NotImplementedError()
    return sample

def from_simple_to_space(
        self, sample: Union[Dict[str, List[Number]], List[Number]]
) -> Union[Dict[str, np.ndarray], np.ndarray]:
    # no information about the numpy data type!?
    raise NotImplementedError()
    return sample

def from_info_to_dict(self, info):
    # no idea
    raise NotImplementedError()
    return info


def from_dict_info(info):
    pass


class RemoteEnvironmentServer(AbstractEnv):

    def __init__(self, env):
        # todo: config contains also complex types: convertable to simple types using benchmark class....
        self.env : env

    def step(self, action: Union[Dict[str, List[Number]], List[Number]]):
        # are action always convertable to list of float / int ?

        action = from_simple_to_space(action)
        state, reward, done, info = self.env.step(action)

        state = from_space_to_simple(state)
        info = from_info_to_dict(info)

        return state, reward, done, info

    def reset(self):
        state = self.env.reset()
        state = self.from_simple_to_space(state)
        return state

    def render(self, mode="human"):
        # ever used?
        pass

    # instance sets and instances
    # close ...


class RemoteEnvironmentClient(AbstractEnv):

    def __init(self, env: RemoteEnvironmentServer):
        self.env = env

    def step(self, action: Union[Dict[str, np.ndarray], np.ndarray]) \
            -> Tuple[Union[Dict[str, np.ndarray], np.ndarray], Number, bool, dict]:
        action = from_space_to_simple(action)
        state, reward, done, info = self.env.step(action)

        state = from_simple_to_space(state)
        info = from_dict_info(info)

        return state, reward, done, info

    def reset(self) -> Union[Dict[str, np.ndarray], np.ndarray]:
        state = self.env.reset()
        state = self.env.reset()
        state = from_simple_to_space(state)
        return state

