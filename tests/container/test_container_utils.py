import json
import unittest

import numpy as np
from icecream import ic

from dacbench.container.container_utils import Encoder, Decoder
from gym.spaces import Box, Discrete, Tuple, MultiDiscrete, MultiBinary, Dict


class TestEncoder(unittest.TestCase):
    def test_spaces(self):
        box = Box(low=-1,     high=1, shape=(2,))
        multi_discrete = MultiDiscrete([[2, 1], [2, 1]])
        multi_binary = MultiBinary([2, 2])
        discrete = Discrete(2)
        spaces = [box, discrete, multi_discrete, multi_binary]

        for space in spaces:
            with self.subTest(msg=str(type(space)), space=space):
                serialized = json.dumps(space, cls=Encoder)
                restored_space = json.loads(serialized, cls=Decoder)
                self.assertEqual(space, restored_space)
                TestEncoder.helper_test_sample(space, restored_space)

    @staticmethod
    def helper_test_sample(space1, space2):
        s1 = space1.sample()
        s2 = space2.sample()
        np.testing.assert_equal(s1, s2)

    def test_recursive_spaces(self):
        tuple_space = Tuple((Box(low=-1, high=1, shape=(2,)), Box(low=-1, high=1, shape=(2,))))
        dict_space = Dict({'a': Box(low=-1, high=1, shape=(2,)), 'b': MultiBinary([2, 2]), 'c': Tuple((Box(low=-1, high=1, shape=(2,)), Box(low=-1, high=1, shape=(2,))))})

        spaces = [tuple_space, dict_space]

        for space in spaces:
            with self.subTest(msg=str(type(space)), space=space):
                serialized = json.dumps(space, cls=Encoder)
                restored_space = json.loads(serialized, cls=Decoder)
                self.assertEqual(space, restored_space)
                TestEncoder.helper_test_sample(space, restored_space)

    def test_random_state(self):
        state = np.random.RandomState(42)
        serialized = json.dumps(state, cls=Encoder)
        restored_state = json.loads(serialized, cls=Decoder)
        np.testing.assert_equal(state.get_state(), restored_state.get_state())
        np.testing.assert_equal(state.random(10), restored_state.random(10))

