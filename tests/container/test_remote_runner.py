import subprocess
import unittest
import signal

from icecream import ic

from dacbench.agents import StaticAgent, RandomAgent
import dacbench.benchmarks
from dacbench.container.remote_runner import RemoteRunner
from dacbench.run_baselines import DISCRETE_ACTIONS

from time import sleep
class TestRemoteRunner(unittest.TestCase):
    def setUp(self) -> None:
        self.name_server_process = subprocess.Popen(
            [
                "pyro4-ns"
            ]
        )
        sleep(1)
        self.daemon_process = subprocess.Popen(
            [
                "python",
                "dacbench/container/remote_runner.py"
            ]
        )
        sleep(1)



    def run_agent_on_benchmark_test(self, benchmark, agent_creation_function):
        remote_runner = RemoteRunner(benchmark)
        agent = agent_creation_function(remote_runner.get_environment())
        remote_runner.run(agent, 1)

    def test_step(self):
        benchmarks = dacbench.benchmarks .__all__[1:]

        for benchmark in benchmarks:
            benchmark_class = getattr(dacbench.benchmarks, benchmark)
            benchmark_instance = benchmark_class()

            action = DISCRETE_ACTIONS[benchmark][0]
            agent_creation_functions = [
                (lambda env: StaticAgent(env, action), f'Static {action}'),
                (lambda env: RandomAgent(env), 'random'),
            ]
            for agent_creation_function, agent_info in agent_creation_functions:
                with self.subTest(msg=f"[Benchmark]{benchmark}, [Agent]{agent_info}", agent_creation_function=agent_creation_function, benchmark=benchmark):
                    ic(benchmark, agent_info)
                    self.run_agent_on_benchmark_test(benchmark_instance, agent_creation_function)




    def tearDown(self) -> None:
        self.name_server_process.send_signal(signal.SIGTERM)
        self.daemon_process.send_signal(signal.SIGTERM)
