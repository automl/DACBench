from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np
import pytest
from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.challenge_benchmarks.reward_quality_challenge.reward_functions import (
    random_reward,
)
from dacbench.challenge_benchmarks.state_space_challenge.random_states import (
    small_random_sigmoid_state,
)
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete


class LessAbstractBenchmark(AbstractBenchmark):
    def get_environment(self):
        pass


class TestAbstractBenchmark(unittest.TestCase):
    def test_setup(self):
        bench = LessAbstractBenchmark()
        assert bench.config is None

    def test_config_file_management(self):
        bench = LessAbstractBenchmark()

        bench.config = objdict({"seed": 0})
        test_config = objdict({"seed": 10})
        with open("test_conf.json", "w+") as fp:
            json.dump(test_config, fp)
        assert bench.config.seed == 0
        bench.read_config_file("test_conf.json")
        assert bench.config.seed == 10
        assert len(bench.config.keys()) == 1
        os.remove("test_conf.json")

        bench.save_config("test_conf2.json")
        with open("test_conf2.json") as fp:
            recovered = json.load(fp)
        assert recovered["seed"] == 10
        assert len(recovered.keys()) == 2
        os.remove("test_conf2.json")

    def test_from_and_to_json(self):
        bench1 = LessAbstractBenchmark(config_path="tests/test_config.json")
        json1 = bench1.serialize_config()
        bench2 = LessAbstractBenchmark(config=objdict(json1))
        json2 = bench2.serialize_config()

        print(json1)
        print(json2)
        assert json1 == json2

    def test_attributes(self):
        bench = LessAbstractBenchmark()
        bench.config = objdict({"seed": 0})
        assert bench.config.seed == bench.config["seed"]
        bench.config.seed = 42
        assert bench.config["seed"] == 42

    def test_getters_and_setters(self):
        bench = LessAbstractBenchmark()
        bench.config = objdict({"seed": 0})
        config = bench.get_config()
        assert issubclass(type(config), dict)

        bench.set_seed(100)
        assert bench.config.seed == 100

        bench.set_action_space("Discrete", [4])
        assert bench.config.action_space == "Discrete"
        assert bench.config.action_space_args == [4]

        bench.set_observation_space("Box", [[1], [0]], float)
        assert bench.config.observation_space == "Box"
        assert bench.config.observation_space_args[0] == [1]
        assert bench.config.observation_space_type == float

    def test_reading_and_saving_config(self):
        bench1 = LessAbstractBenchmark(config_path="tests/test_config.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "config.json")
            bench1.save_config(config_file)

            bench2 = LessAbstractBenchmark()
            bench2.read_config_file(config_file)

            assert bench1.config["state_method"] == bench2.config["state_method"]
            assert bench1.config["state_method"] == small_random_sigmoid_state

            assert bench1.config["reward_function"] == bench2.config["reward_function"]
            assert bench1.config["reward_function"] == random_reward

            assert bench1.jsonify_wrappers() == bench2.jsonify_wrappers()
            assert bench1.jsonify_wrappers() == [["RewardNoiseWrapper", []]]

    def test_jsonify_wrappers_and_dejson_wrappers(self):
        bench = LessAbstractBenchmark()
        empty_warpper_list = bench.jsonify_wrappers()
        assert empty_warpper_list == []

    def test_space_to_list_and_list_to_space(self):
        def assert_restorable(space):
            space_restored = bench.list_to_space(bench.space_to_list(space))
            assert space == space_restored

        bench = LessAbstractBenchmark()

        space = Box(
            low=np.array([0, 0]),
            high=np.array([1, 1]),
        )
        assert_restorable(space)

        space = Discrete(2)
        assert_restorable(space)

        space = Dict(
            {
                "box": Box(
                    low=np.array([0, 0]),
                    high=np.array([1, 1]),
                ),
                "discrete": Discrete(n=2),
            }
        )

        assert_restorable(space)

        space = MultiDiscrete([2, 3])
        assert_restorable(space)

        space = MultiBinary(3)
        assert_restorable(space)

    def test_objdict(self):
        d = objdict({"dummy": 0})

        assert d["dummy"] == d.dummy
        with pytest.raises(KeyError):
            d["error"]
        with pytest.raises(AttributeError):
            d.error

        d["error"] = 12
        assert d.error == 12
        del d.error
        assert "error" not in d

        with pytest.raises(KeyError):
            del d["error"]
        with pytest.raises(AttributeError):
            del d.error

    def test_objdict_equal(self):
        assert objdict({"dummy": 0}) == objdict({"dummy": 0})
        assert objdict({"dummy": np.array([1, 2])}) == objdict(
            {"dummy": np.array([1, 2])}
        )

        assert objdict({"dummy": np.array([1, 2])}) != objdict(
            {"dummy": np.array([1, 0])}
        )
        assert objdict({"dummy": np.array([1, 2])}) != objdict({"dummy": 0})
