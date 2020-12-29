import json
import unittest
from pathlib import Path
import tempfile

from dacbench.agents.simple_agents import RandomAgent
from dacbench.benchmarks import SigmoidBenchmark
from dacbench.logger import ModuleLogger, Logger


class TestLogger(unittest.TestCase):
    def test_wrapper(self):
        episodes = 3
        # todo instances
        temp_dir = tempfile.TemporaryDirectory()
        experiment_name = "test_experiment"
        logger = Logger(
            output_path=Path(temp_dir.name), experiment_name=experiment_name
        )

        benchmark = SigmoidBenchmark()
        env = benchmark.get_benchmark()
        agent = RandomAgent(env)

        env_logger = logger.add_env(env)

        for episode in range(1, episodes + 1):
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

                step += 1
            agent.end_episode(state, reward)
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
        temp_dir.cleanup()


class TestModuleLogger(unittest.TestCase):
    def test_basic_logging(self):
        temp_dir = tempfile.TemporaryDirectory()
        experiment_name = "test_experiment"
        module_name = "module"
        logger = ModuleLogger(
            output_path=Path(temp_dir.name),
            experiment_name=experiment_name,
            module=module_name,
        )

        for episode in range(10):
            logger.log("episode_logged", episode)
            for step in range(3):
                logger.log("step_logged", step)
                logger.next_step()
            logger.next_episode()
        logger.write()

        with open(logger.log_file.name, "r") as log_file:
            logs = list(map(json.loads, log_file))
        self.assertEqual(
            10 * 3, len(logs), "For each step with logging done in it one line exit"
        )
        for log in logs:
            self.assertTrue(
                all(
                    log["episode"] == logged_episode
                    for logged_episode in log.get("episode_logged", {"values": []})[
                        "values"
                    ]
                )
            )
            self.assertTrue(
                all(
                    log["step"] == logged_episode
                    for logged_episode in log.get("step_logged", {"values": []})[
                        "values"
                    ]
                )
            )

        temp_dir.cleanup()
