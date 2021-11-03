import subprocess
import unittest
import signal

from dacbench.agents import StaticAgent
from dacbench.benchmarks import SigmoidBenchmark
from dacbench.container.remote_runner import RemoteRunner


class TestRemoteRunner(unittest.TestCase):
    def setUp(self) -> None:
        self.name_server_process = subprocess.Popen(
            [
                "pyro4-ns"
            ]
        )

        self.daemon_process = subprocess.Popen(
            [
                "python",
                "dacbench/container/remote_runner.py"
            ]
        )

    def test_step(self):
        benchmark = SigmoidBenchmark()
        agent = StaticAgent(env=None, action=0)
        server_uri = f"PYRONAME:RemoteRunnerServer"

        remote_runner = RemoteRunner(benchmark, server_uri)
        remote_runner.run(agent, 1, 42)

    def tearDown(self) -> None:
        self.name_server_process.send_signal(signal.SIGTERM)
        self.daemon_process.send_signal(signal.SIGTERM)