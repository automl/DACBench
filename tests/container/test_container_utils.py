from __future__ import annotations

import json
import unittest

from dacbench.container.container_utils import Decoder, Encoder
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Tuple


class TestEncoder(unittest.TestCase):
    def test_spaces(self):
        box = Box(low=-1, high=1, shape=(2,))
        multi_discrete = MultiDiscrete([[2, 1], [2, 1]])
        multi_binary = MultiBinary([2, 2])
        discrete = Discrete(2)
        spaces = [box, discrete, multi_discrete, multi_binary]

        for space in spaces:
            with self.subTest(msg=str(type(space)), space=space):
                serialized = json.dumps(space, cls=Encoder)
                restored_space = json.loads(serialized, cls=Decoder)
                assert space == restored_space

    def test_recursive_spaces(self):
        tuple_space = Tuple(
            (Box(low=-1, high=1, shape=(2,)), Box(low=-1, high=1, shape=(2,)))
        )
        dict_space = Dict(
            {
                "a": Box(low=-1, high=1, shape=(2,)),
                "b": MultiBinary([2, 2]),
                "c": Tuple(
                    (Box(low=-1, high=1, shape=(2,)), Box(low=-1, high=1, shape=(2,)))
                ),
            }
        )

        spaces = [tuple_space, dict_space]

        for space in spaces:
            with self.subTest(msg=str(type(space)), space=space):
                serialized = json.dumps(space, cls=Encoder)
                restored_space = json.loads(serialized, cls=Decoder)
                assert space == restored_space
