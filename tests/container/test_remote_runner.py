import logging
import signal
import subprocess
import unittest

import Pyro4

import dacbench.benchmarks
from dacbench.agents import StaticAgent, RandomAgent
from dacbench.container.container_utils import wait_for_unixsocket
from dacbench.container.remote_runner import RemoteRunner
from dacbench.run_baselines import DISCRETE_ACTIONS


class TestRemoteRunnerServerFactory(unittest.TestCase):
    def setUp(self) -> None:
        self.socket_id = RemoteRunner.id_generator()
        self.socket = RemoteRunner.socket_from_id(self.socket_id)

        self.daemon_process = subprocess.Popen(
            [
                "python",
                "dacbench/container/remote_runner.py",
                "-u",
                str(self.socket)
            ]
        )

        wait_for_unixsocket(self.socket)

    def test_running(self):
        ns = Pyro4.Proxy(f"PYRO:Pyro.NameServer@./u:{self.socket}")
        factory_uri = ns.lookup(RemoteRunner.FACTORY_NAME)
        factory = Pyro4.Proxy(factory_uri)

        remote_runner = factory.create()

    def tearDown(self) -> None:
        self.daemon_process.send_signal(signal.SIGINT)


class TestRemoteRunner(unittest.TestCase):

    def run_agent_on_benchmark_test(self, benchmark, agent_creation_function):
        remote_runner = RemoteRunner(benchmark)
        agent = agent_creation_function(remote_runner.get_environment())
        remote_runner.run(agent, 1)

    def test_step(self):
        skip_benchmarks = ['CMAESBenchmark', 'LubyBenchmark']
        benchmarks = ["SGDBenchmark"]  # dacbench.benchmarks.__all__

        for benchmark in benchmarks:
            if benchmark in skip_benchmarks:
                continue
                # todo Skipping since serialization is not done yet. https://github.com/automl/DACBench/issues/107
                # self.skipTest(reason="Skipping since serialization is not done yet. https://github.com/automl/DACBench/issues/107")
            if benchmark not in DISCRETE_ACTIONS:
                logging.warning(f"Skipping test for {benchmark} since no discrete actions are available")
                continue
            benchmark_class = getattr(dacbench.benchmarks, benchmark)
            benchmark_instance = benchmark_class()

            action = DISCRETE_ACTIONS[benchmark][0]
            agent_creation_functions = [
                (lambda env: StaticAgent(env, action), f'Static {action}'),
                (lambda env: RandomAgent(env), 'random'),
            ]
            for agent_creation_function, agent_info in agent_creation_functions:
                with self.subTest(msg=f"[Benchmark]{benchmark}, [Agent]{agent_info}", agent_creation_function=agent_creation_function, benchmark=benchmark):
                    self.run_agent_on_benchmark_test(benchmark_instance, agent_creation_function)
