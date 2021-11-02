import os

os.environ["PYRO_LOGFILE"] = "stderr"
os.environ["PYRO_LOGLEVEL"] = "DEBUG"

from dacbench.abstract_agent import AbstractDACBenchAgent
from dacbench.abstract_benchmark import objdict
import Pyro4

from dacbench.container.remote_env import RemoteEnvironmentServer


@Pyro4.expose
class RemoteRunnerServer:
    def __init__(self, pyro_demon):
        self.benchmark = None
        self.pyro_demon = pyro_demon

    def start(self,  config : objdict, benchmark : type):
       self.benchmark = benchmark(config=config)

    def get_env(self, seed) -> str:
        env = self.benchmark.get_env(seed)

        # set up logger and stuff

        self.env = RemoteEnvironmentServer(env)
        uri = self.pyro_demon.register(self.env)
        return uri

class RemoteRunner:
    def __init__(self, config : objdict, benchmark : type, remote_runner_uri : str):
        # todo: implement
        # todo switch to json config?
        # load container
        # start container and server
        # connect to continaer and create RemoveBenchmarkClient object
        # copy from AbstractBenchmarkClient

        self.remote_runner: RemoteRunnerServer = Pyro4.Proxy(remote_runner_uri)
        self.remote_runner.start(config, benchmark)


    def run(self, agent : AbstractDACBenchAgent, number_of_episodes : int, seed : int = None):
        # todo: agent often needs env for creation ...
        env_uri = self.remote_runner.get_env(seed)

        env = Pyro4.Proxy(env_uri)
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