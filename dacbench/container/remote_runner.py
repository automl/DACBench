"""This is strongly guided and partially copy from:https://github.com/automl/HPOBench/blob/master/hpobench/container/client_abstract_benchmark.py."""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid1

import Pyro4
import Pyro4.naming

from dacbench.abstract_benchmark import AbstractBenchmark
from dacbench.argument_parsing import PathType
from dacbench.container.container_utils import wait_for_unixsocket
from dacbench.container.remote_env import (
    RemoteEnvironmentClient,
    RemoteEnvironmentServer,
)

if TYPE_CHECKING:
    from dacbench.abstract_agent import AbstractDACBenchAgent

# Needed in order to combine event loops of name_server and daemon
Pyro4.config.SERVERTYPE = "multiplex"

# Read in the verbosity level from the environment variable
log_level_str = os.environ.get("DACBENCH_DEBUG", "false")

LOG_LEVEL = logging.INFO
LOG_LEVEL = logging.DEBUG if log_level_str == "true" else logging.INFO

root = logging.getLogger()
root.setLevel(level=LOG_LEVEL)

logger = logging.getLogger(__name__)
logger.setLevel(level=LOG_LEVEL)

# This option improves the quality of stacktraces if a container crashes
sys.excepthook = Pyro4.util.excepthook
# os.environ["PYRO_LOGFILE"] = "pyro.log"
# os.environ["PYRO_LOGLEVEL"] = "DEBUG"

# Number of tries to connect to server
MAX_TRIES = 5

SOCKET_PATH = Path("/tmp/dacbench/sockets")  # noqa: S108


@Pyro4.expose
class RemoteRunnerServer:
    """Server for container running."""

    def __init__(self, pyro_demon):
        """Init server."""
        self.benchmark = None
        self.pyro_demon = pyro_demon

    def start(self, config: str, benchmark: tuple[str, str]):
        """Start server."""
        benchmark = AbstractBenchmark.import_from(*benchmark)

        self.benchmark = benchmark.from_json(config)

    def get_environment(self) -> str:
        """Get environment."""
        env = self.benchmark.get_environment()

        # set up logger and stuff

        self.env = RemoteEnvironmentServer(env)
        return self.pyro_demon.register(self.env)


class RemoteRunner:
    """Runner for remote benchmarks."""

    FACTORY_NAME: str = "RemoteRunnerServerFactory"

    def __init__(
        self,
        benchmark: AbstractBenchmark,
        container_name: str | None = None,
        container_source: str | None = None,
        container_tag: str = "latest",
        env_str: str | None = "",
        bind_str: str | None = "",
        gpu: bool | None = False,
        socket_id=None,
    ):
        """Runner for containers.

        Parameters
        ----------
        benchmark: AbstractBenchmark
            The benchmark to run
        container_name : str
            name for container
        container_source : Optional[str]
            Path to the container. Either local path or url to a hosting platform,
            e.g. singularity hub.
        container_tag : str
            Singularity containers are specified by an address as well as a
            container tag. We use the tag as a version number. By default the tag is
            set to `latest`, which then pulls the latest container from the container
            source. The tag-versioning allows the users to rerun an experiment, which
            was performed with an older container version. Take a look in the
            container_source to find the right tag to use.
        bind_str : Optional[str]
            Defaults to ''. You can bind further directories into the container.
            This string have the form src[:dest[:opts]].
            For more information, see https://sylabs.io/guides/3.5/user-guide/bind_paths_and_mounts.html
        env_str : Optional[str]
            Defaults to ''. Sometimes you want to pass a parameter to your container.
            You can do this by setting some environmental variables.
            The list should follow the form VAR1=VALUE1,VAR2=VALUE2,..
            For more information, see
            https://sylabs.io/guides/3.5/user-guide/environment_and_metadata.html#environment-overview
        gpu : bool
            If True, the container has access to the local cuda-drivers. (Not tested)
        socket_id : Optional[str]
            Setting up the container is done in two steps:
            1) Start the benchmark on a random generated socket id.
            2) Create a proxy connection to the container via this socket id.

            When no `socket_id` is given, a new container is started.
            The `socket_id` (address) of this containers is
            stored in the class attribute Benchmark.socket_id

            When a `socket_id` is given, instead of creating a new container,
            connect only to the container that is reachable at `socket_id`.
            Make sure that a container is already running with the address `socket_id`.

        """
        logger.info(f"Logging level: {logger.level}")
        # connect to already running server if a socket_id is given.
        # In this case, skip the init of the benchmark
        self.__proxy_only = socket_id is not None
        self.__socket_path = SOCKET_PATH

        if not self.__proxy_only:
            self.__socket_id = self.id_generator()
            # todo for now only work with given container source (local)
            self.load_benchmark(
                benchmark=benchmark,
                container_name=container_name,
                container_source=container_source,
                container_tag=container_tag,
            )
            self.__start_server(env_str=env_str, bind_str=bind_str, gpu=gpu)
        else:
            self.__socket_id = socket_id

        self.__connect_to_server(benchmark)

    @property
    def socket(self) -> Path:
        """Get socket."""
        return self.socket_from_id(self.__socket_id)

    @staticmethod
    def id_generator() -> str:
        """Helper function: Creates unique socket ids for the benchmark server."""
        return str(uuid1())

    @staticmethod
    def socket_from_id(socket_id: str) -> Path:
        """Get socket from id."""
        return Path(SOCKET_PATH) / f"{socket_id}.unixsock"

    def __start_server(self, env_str, bind_str, gpu):
        """Starts container and the pyro server.

        Parameters
        ----------
        env_str : str
            Environment string for the container
        bind_str : str
            Bind string for the container
        gpu : bool
            True if the container should use gpu, False otherwise

        """
        # start container
        logger.debug(f"Starting server on {self.socket}")

        # todo add mechanism to to retry if failing
        self.daemon_process = subprocess.Popen(
            [  # noqa: S603, S607
                "singularity",
                "run",
                "-e",
                str(self.container_source),
                "-u",
                str(self.socket),
            ]
        )

        # todo should be configurable
        wait_for_unixsocket(self.socket, 10)

    def __connect_to_server(self, benchmark: AbstractBenchmark):
        """Connects to the server and initializes the benchmark."""
        # Pyro4.config.REQUIRE_EXPOSE = False
        # Generate Pyro 4 URI for connecting to client
        ns = Pyro4.Proxy(f"PYRO:Pyro.NameServer@./u:{self.socket}")
        factory_uri = ns.lookup(self.FACTORY_NAME)

        factory = Pyro4.Proxy(factory_uri)
        remote_runner_uri = factory.create()
        self.remote_runner: RemoteRunnerServer = Pyro4.Proxy(remote_runner_uri)

        serialized_config = benchmark.to_json()
        serialized_type = benchmark.class_to_str()
        self.remote_runner.start(serialized_config, serialized_type)
        self.env = None

    def get_environment(self):
        """Get remote environment."""
        if self.env is None:
            env_uri = self.remote_runner.get_environment()
            remote_env_server = Pyro4.Proxy(env_uri)
            self.env = RemoteEnvironmentClient(remote_env_server)
        return self.env

    def run(self, agent: AbstractDACBenchAgent, number_of_episodes: int):
        """Run agent on remote."""
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

    def close(self):
        """Termiante all processes."""
        # todo add context manager
        self.daemon_process.terminate()
        self.daemon_process.wait()

    def __del__(self):
        """Close."""
        self.close()

    def load_benchmark(
        self,
        benchmark: AbstractBenchmark,
        container_name: str,
        container_source: str | Path,
        container_tag: str,
    ):
        """Load benchmark from recipe."""
        # see for implementation guideline hpobench
        # hpobench/container/client_abstract_benchmark.py
        # in the end self.container_source should contain the path to the file to run

        logger.warning("Only container source is used")
        container_source = (
            container_source
            if isinstance(container_source, Path)
            else Path(container_source)
        )

        self.container_source = container_source


@Pyro4.expose
class RemoteRunnerServerFactory:
    """Creates remoter runner servers."""

    def __init__(self, pyro_demon):
        """Make server factory."""
        self.pyro_demon = pyro_demon

    def create(self):
        """Get server."""
        remote_runner_server = RemoteRunnerServer(pyro_demon=self.pyro_demon)
        return daemon.register(remote_runner_server)

    def __call__(self):
        """Make."""
        return self.create()


if __name__ == "__main__":
    # todo refactor move to RemoverRunnerServer
    parser = argparse.ArgumentParser(
        description="Runs the benchmark remote server inside a container"
    )

    parser.add_argument(
        "--unixsocket",
        "-u",
        type=PathType(exists=False, type="socket"),
        required=False,
        default=None,
        dest="socket",
        help=(
            "The path to a exiting socket to run the name server on. "
            "If none a new socket unixsocket is created."
        ),
    )

    args = parser.parse_args()

    daemon_socket = RemoteRunner.socket_from_id(RemoteRunner.id_generator())
    ns_socket = (
        args.socket
        if args.socket
        else RemoteRunner.socket_from_id(RemoteRunner.id_generator())
    )
    print(ns_socket)
    daemon_socket.parent.mkdir(parents=True, exist_ok=True)
    ns_socket.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Starting Pyro4 Nameserver on {ns_socket} and Pyro4 Daemon on {daemon_socket}"
    )
    name_server_uir, name_server_daemon, _ = Pyro4.naming.startNS(
        unixsocket=str(ns_socket)
    )
    daemon = Pyro4.Daemon(unixsocket=str(daemon_socket))
    daemon.combine(name_server_daemon)
    factory = RemoteRunnerServerFactory(daemon)
    factory_uri = daemon.register(factory)
    name_server_daemon.nameserver.register("RemoteRunnerServerFactory", factory_uri)

    daemon.requestLoop()

    daemon_socket.unlink()
    ns_socket.unlink()
