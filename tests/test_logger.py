import json
import unittest
from pathlib import Path
import tempfile
from dacbench.logger import Logger


class TestLogger(unittest.TestCase):
    def test_basic_logging(self):
        temp_dir = tempfile.TemporaryDirectory()
        experiment_name = "test_experiment"
        logger = Logger(
            output_path=Path(temp_dir.name), experiment_name=experiment_name
        )

        for episode in range(10):
            logger.log("episode_logged", episode)
            for step in range(3):
                logger.log("step_logged", step)
                logger.next_step()
            logger.next_episode()
        logger.write()

        logger.log_file.seek(0)
        logs = logger.log_file.read().splitlines()
        self.assertEqual(
            10 * 3, len(logs), "For each step with logging done in it one line exit"
        )
        for line in logs:
            log_step = json.loads(line)
            self.assertTrue(
                all(
                    log_step["episode"] == logged_episode
                    for logged_episode in log_step.get(
                        "episode_logged", {"values": []}
                    )["values"]
                )
            )
            self.assertTrue(
                all(
                    log_step["step"] == logged_episode
                    for logged_episode in log_step.get("step_logged", {"values": []})[
                        "values"
                    ]
                )
            )

        temp_dir.cleanup()
