import json
import unittest
from pathlib import Path
import tempfile

from dacbench.agents.simple_agents import RandomAgent
from dacbench.benchmarks import SigmoidBenchmark
from dacbench.logger import ModuleLogger, Logger


class TestLogger(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_env(self):
        episodes = 3
        # todo instances
        experiment_name = "test_env"
        logger = Logger(
            output_path=Path(self.temp_dir.name), experiment_name=experiment_name
        )

        benchmark = SigmoidBenchmark()
        env = benchmark.get_benchmark()
        agent = RandomAgent(env)

        env_logger = logger.add_module(env)

        for episode in range(episodes):
            state = env.reset()
            done = False
            reward = 0
            step = 0
            while not done:
                action = agent.act(state, reward)
                env_logger.log("logged_step", step)
                env_logger.log("logged_episode", episode)
                next_state, reward, done, _ = env.step(action)
                env_logger.log("reward", reward)
                env_logger.log("done", done)
                agent.train(next_state, reward)
                state = next_state
                logger.next_step()

                step += 1
            agent.end_episode(state, reward)
            logger.next_episode()
            logger.write()

        env.close()
        logger.close()

        with open(env_logger.log_file.name, "r") as log_file:
            logs = list(map(json.loads, log_file))

        for log in logs:
            if "logged_step" in log:
                self.assertEqual(log["logged_step"]["values"][0], log["step"])
            if "logged_episode" in log:
                self.assertEqual(log["logged_episode"]["values"][0], log["episode"])


class TestModuleLogger(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_basic_logging(self):
        experiment_name = "test_basic_logging"
        module_name = "module"
        episodes = 10
        steps = 3

        logger = ModuleLogger(
            output_path=Path(self.temp_dir.name),
            experiment_name=experiment_name,
            module=module_name,
        )

        for episode in range(episodes):
            logger.log("episode_logged", episode)
            for step in range(steps):
                logger.log("step_logged", step)
                logger.next_step()
            logger.next_episode()
        logger.write()

        with open(logger.log_file.name, "r") as log_file:
            logs = list(map(json.loads, log_file))

        self.assertEqual(
            episodes * steps,
            len(logs),
            "For each step with logging done in it one line exit",
        )

        for log in logs:
            if "logged_step" in log:
                self.assertTrue(
                    all(
                        log["step"] == logged_step
                        for logged_step in log["logged_step"]["values"]
                    ),
                )
            if "logged_episode" in log:
                self.assertTrue(
                    all(
                        log["step"] == logged_episode
                        for logged_episode in log["episode_logged"]["values"]
                    ),
                )
