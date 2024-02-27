from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from dacbench.agents.simple_agents import RandomAgent
from dacbench.benchmarks import SigmoidBenchmark
from dacbench.logger import Logger, ModuleLogger, log2dataframe
from gymnasium import spaces
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete


class TestLogger(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

        episodes = 80
        seeds = [0, 1, 3, 4, 5]
        experiment_name = "test_env"
        logger = Logger(
            output_path=Path(self.temp_dir.name),
            experiment_name=experiment_name,
            step_write_frequency=None,
            episode_write_frequency=None,
        )

        benchmark = SigmoidBenchmark()
        env = benchmark.get_benchmark()
        agent = RandomAgent(env)
        logger.set_env(env)

        env_logger = logger.add_module(env)
        for seed in seeds:
            env.seed(seed)
            logger.reset_episode()

            for episode in range(episodes):
                state, _ = env.reset()
                terminated, truncated = False, False
                reward = 0
                step = 0
                while not (terminated or truncated):
                    action = agent.act(state, reward)
                    env_logger.log(
                        "logged_step",
                        step,
                    )
                    env_logger.log("logged_seed", env.initial_seed)

                    env_logger.log("logged_instance", env.get_inst_id())

                    env_logger.log(
                        "logged_episode",
                        episode,
                    )
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    env_logger.log(
                        "reward",
                        reward,
                    )
                    env_logger.log(
                        "terminated",
                        terminated,
                    )
                    env_logger.log(
                        "truncated",
                        truncated,
                    )
                    agent.train(next_state, reward)
                    state = next_state
                    logger.next_step()

                    step += 1
                agent.end_episode(state, reward)
                logger.next_episode()

        env.close()
        logger.close()

        self.log_file = env_logger.log_file.name

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_env_logger(self):
        with open(self.log_file) as log_file:
            logs = list(map(json.loads, log_file))

        for log in logs:
            # todo check when nan occurs
            if "logged_step" in log:
                assert log["logged_step"]["values"][0] == log["step"]
            if "logged_episode" in log:
                assert log["logged_episode"]["values"][0] == log["episode"]
            # check of only one seed occurs per episode
            seeds = set(log["logged_seed"]["values"])
            assert len(seeds) == 1
            (seed,) = seeds
            assert seed == log["seed"]

            # check of only one instance occurs per episode
            instances = set(log["logged_instance"]["values"])
            assert len(seeds) == 1
            (instance,) = instances
            assert instance == log["instance"]

    def test_data_loading(self):
        with open(self.log_file) as log_file:
            logs = list(map(json.loads, log_file))

        dataframe = log2dataframe(
            logs,
            wide=True,
        )
        assert (dataframe.logged_step == dataframe.step).all()
        assert (dataframe.logged_episode == dataframe.episode).all()


class TestModuleLogger(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_spaces(self):
        experiment_name = "test_spaces"
        module_name = "module"

        logger = ModuleLogger(
            output_path=Path(self.temp_dir.name),
            experiment_name=experiment_name,
            module=module_name,
            step_write_frequency=None,
            episode_write_frequency=None,
        )
        seed = 3

        # Discrete
        space = Discrete(n=3)
        space.seed(seed)
        logger.log_space("Discrete", space.sample())

        # MultiDiscrete
        space = MultiDiscrete(np.array([3, 2]))
        space.seed(seed)
        logger.log_space("MultiDiscrete", space.sample())

        # Dict
        space = Dict(
            {
                "predictiveChangeVarDiscountedAverage": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,)
                ),
                "predictiveChangeVarUncertainty": spaces.Box(
                    low=0, high=np.inf, shape=(1,)
                ),
                "lossVarDiscountedAverage": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,)
                ),
                "lossVarUncertainty": spaces.Box(low=0, high=np.inf, shape=(1,)),
                "currentLR": spaces.Box(low=0, high=1, shape=(1,)),
                "trainingLoss": spaces.Box(low=0, high=np.inf, shape=(1,)),
                "validationLoss": spaces.Box(low=0, high=np.inf, shape=(1,)),
            }
        )
        space.seed(seed)
        logger.log_space("Dict", space.sample())

        space = Box(np.array([0, 0]), np.array([2, 2]))
        space.seed(seed)
        logger.log_space("Box", space.sample())
        logger.close()

        with open(logger.get_logfile()) as log_file:
            logs = list(map(json.loads, log_file))

        wide = log2dataframe(logs, wide=True)
        long = log2dataframe(logs, drop_columns=None)

        assert len(wide) == 1
        first_row = wide.iloc[0]

        # Discrete
        assert not np.isnan(first_row.Discrete)

        # MultiDiscrete
        assert not np.isnan(first_row.MultiDiscrete_0)
        assert not np.isnan(first_row.MultiDiscrete_1)
        simultaneous_logged = long[
            (long.name == "MultiDiscrete_0") | (long.name == "MultiDiscrete_1")
        ]
        assert len(simultaneous_logged.time.unique()) == 1

        # Dict
        expected_columns = [
            "Dict_currentLR_0",
            "Dict_lossVarDiscountedAverage_0",
            "Dict_lossVarUncertainty_0",
            "Dict_predictiveChangeVarDiscountedAverage_0",
            "Dict_predictiveChangeVarUncertainty_0",
            "Dict_trainingLoss_0",
        ]

        for expected_column in expected_columns:
            assert not np.isnan(first_row[expected_column])

        simultaneous_logged = long[long.name.isin(expected_columns)]
        assert len(simultaneous_logged.time.unique()) == 1

        # Box
        assert not np.isnan(first_row.Box_0)
        assert not np.isnan(first_row.Box_1)

        simultaneous_logged = long[(long.name == "Box_0") | (long.name == "Box_1")]
        assert len(simultaneous_logged.time.unique()) == 1

    def test_log_numpy(self):
        experiment_name = "test_log_numpy"
        module_name = "module"

        logger = ModuleLogger(
            output_path=Path(self.temp_dir.name),
            experiment_name=experiment_name,
            module=module_name,
            step_write_frequency=None,
            episode_write_frequency=None,
        )

        logger.log(
            "state",
            np.array([1, 2, 3]),
        )
        logger.close()

        with open(logger.get_logfile()) as log_file:
            logs = list(map(json.loads, log_file))

        dataframe = log2dataframe(logs, wide=True)
        assert dataframe.iloc[0].state == (1, 2, 3)

    def test_numpy_logging(self):
        experiment_name = "test_numpy_logging"
        module_name = "module"
        logger = ModuleLogger(
            output_path=Path(self.temp_dir.name),
            experiment_name=experiment_name,
            module=module_name,
            step_write_frequency=None,
            episode_write_frequency=None,
        )

        logger.set_additional_info(np=np.zeros((2, 3, 3)))
        logger.log("test", 0)

        logger.close()

        with open(logger.get_logfile()) as log_file:
            logs = list(map(json.loads, log_file))

        dataframe = log2dataframe(logs, wide=True)

        expected_result = (((0,) * 3,) * 3,) * 2
        assert dataframe.iloc[0].np == expected_result

    def test_basic_logging(self):
        experiment_name = "test_basic_logging"
        module_name = "module"
        episodes = 10
        steps = 3

        logger = ModuleLogger(
            output_path=Path(self.temp_dir.name),
            experiment_name=experiment_name,
            module=module_name,
            step_write_frequency=None,
            episode_write_frequency=None,
        )

        for episode in range(episodes):
            logger.log(
                "episode_logged",
                episode,
            )
            for step in range(steps):
                logger.log(
                    "step_logged",
                    step,
                )
                logger.next_step()
            logger.next_episode()

        logger.close()  # or logger write

        with open(logger.log_file.name) as log_file:
            logs = list(map(json.loads, log_file))

        assert episodes * steps == len(
            logs
        ), "For each step with logging done in it one line exit"

        for log in logs:
            if "logged_step" in log:
                assert all(
                    log["step"] == logged_step
                    for logged_step in log["logged_step"]["values"]
                )
            if "logged_episode" in log:
                assert all(
                    log["step"] == logged_episode
                    for logged_episode in log["episode_logged"]["values"]
                )
