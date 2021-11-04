import os
from typing import Tuple

from icecream import ic

os.environ["PYRO_LOGFILE"] = "pyro.log"
os.environ["PYRO_LOGLEVEL"] = "DEBUG"

from dacbench.abstract_agent import AbstractDACBenchAgent
from dacbench.abstract_benchmark import objdict, AbstractBenchmark
import Pyro4

from dacbench.container.remote_env import RemoteEnvironmentServer, RemoteEnvironmentClient


@Pyro4.expose
class RemoteRunnerServer:
    def __init__(self, pyro_demon):
        self.benchmark = None
        self.pyro_demon = pyro_demon

    def start(self,  config : str, benchmark : Tuple[str, str]):
        benchmark = AbstractBenchmark.import_from(*benchmark)

        self.benchmark = benchmark.from_json(config)

    def get_environment(self) -> str:
        ic(self.benchmark)

        env = self.benchmark.get_environment()
        ic(env)
        # set up logger and stuff

        self.env = RemoteEnvironmentServer(env)
        uri = self.pyro_demon.register(self.env)
        return uri

class RemoteRunner:
    def __init__(self, benchmark : AbstractBenchmark, factory_uri : str = "PYRONAME:RemoteRunnerServerFactory"):
        # load container
        # start container and server
        # connect to continaer and create RemoveBenchmarkClient object
        # copy from AbstractBenchmarkClient

        factory = Pyro4.Proxy(factory_uri)
        remote_runner_uri = factory.create()
        self.remote_runner: RemoteRunnerServer = Pyro4.Proxy(remote_runner_uri)

        serialized_config = benchmark.to_json()
        serialized_type = benchmark.class_to_str()
        ic(serialized_config)
        self.remote_runner.start(serialized_config, serialized_type)
        self.env = None

    def get_environment(self):
        if self.env is None:
            env_uri = self.remote_runner.get_environment()
            remote_env_server = Pyro4.Proxy(env_uri)
            self.env = RemoteEnvironmentClient(remote_env_server)
        return self.env

    def run(self, agent : AbstractDACBenchAgent, number_of_episodes : int):
        # todo: seeding
        env = self.get_environment()

        for _ in range(number_of_episodes):
            state = env.reset()
            done = False
            reward = 0
            while not done:
                action = agent.act(state, reward)
                next_state, reward, done, _ = env.step(action)
                agent.train(next_state, reward)
                state = next_state
            agent.end_episode(state, reward)

        env.close()
        self.env = None

@Pyro4.expose
class RemoteRunnerServerFactory:
    def __init__(self, pyro_demon):
        self.pyro_demon = pyro_demon

    def create(self):
        remote_runner_server = RemoteRunnerServer(pyro_demon=self.pyro_demon)
        remote_runner_server_uri = daemon.register(remote_runner_server)
        return remote_runner_server_uri

    def __call__(self):
        return self.create()

if __name__ == '__main__':

    PORT = 8888
    HOST = "localhost" # add arguments
    name_server = Pyro4.locateNS()
    daemon = Pyro4.Daemon(HOST, PORT)
    factory = RemoteRunnerServerFactory(daemon)
    factory_uri = daemon.register(factory)
    name_server.register("RemoteRunnerServerFactory", factory_uri)

    daemon.requestLoop()