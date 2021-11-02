import os
from typing import Tuple

os.environ["PYRO_LOGFILE"] = "stderr"
os.environ["PYRO_LOGLEVEL"] = "DEBUG"

from dacbench.abstract_agent import AbstractDACBenchAgent
from dacbench.abstract_benchmark import objdict, AbstractBenchmark
import Pyro4

from dacbench.container.remote_env import RemoteEnvironmentServer, RemoteEnvironmentClient

from icecream import ic
@Pyro4.expose
class RemoteRunnerServer:
    def __init__(self, pyro_demon):
        self.benchmark = None
        self.pyro_demon = pyro_demon

    def start(self,  config : str, benchmark : Tuple[str, str]):
        benchmark = AbstractBenchmark.import_from(*benchmark)

        self.benchmark = benchmark.from_json(config)

    def get_environment(self) -> str:


        env = self.benchmark.get_environment()

        # set up logger and stuff

        self.env = RemoteEnvironmentServer(env)
        uri = self.pyro_demon.register(self.env)
        return uri

class RemoteRunner:
    def __init__(self, benchmark : AbstractBenchmark, remote_runner_uri : str):
        # load container
        # start container and server
        # connect to continaer and create RemoveBenchmarkClient object
        # copy from AbstractBenchmarkClient

        self.remote_runner: RemoteRunnerServer = Pyro4.Proxy(remote_runner_uri)

        serialized_config = benchmark.to_json()
        serialized_type = benchmark.class_to_str()
        self.remote_runner.start(serialized_config, serialized_type)

    def __get_environment(self):
        env_uri = self.remote_runner.get_environment()
        remote_env_server = Pyro4.Proxy(env_uri)
        remote_env = RemoteEnvironmentClient(remote_env_server)
        return remote_env

    def run(self, agent : AbstractDACBenchAgent, number_of_episodes : int, seed : int = None):
        # todo: agent often needs env for creation ...
        # todo: seeding
        env = self.__get_environment()

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

if __name__ == '__main__':

    PORT = 8888
    HOST = "localhost" # add arguments
    name_server = Pyro4.locateNS()
    daemon = Pyro4.Daemon(HOST, PORT)
    remote_runner_server = RemoteRunnerServer(pyro_demon=daemon)
    remote_runner_server_uri = daemon.register(remote_runner_server)
    name_server.register('RemoteRunnerServer', remote_runner_server_uri)
    print(remote_runner_server_uri)
    print("Server is up and running")

    daemon.requestLoop()