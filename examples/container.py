"""Container Example."""

from pathlib import Path

# in order to run this we need to build the container first by running
# `singularity build --fakeroot dachbench.sif dacbench/container/singularity_recipes/dachbench.def` from project root.
# For more details refer to dacbench/container/Container Roadmap.md
from dacbench.agents import RandomAgent
from dacbench.benchmarks import FunctionApproximationBenchmark
from dacbench.container.remote_runner import RemoteRunner

if __name__ == "__main__":
    container_source = (Path(__file__).parent.parent / "dacbench.sif").resolve()

    if not container_source.exists():
        raise RuntimeError(
            f"Container file not found ({container_source}). Please build before running this example"
        )

    # config
    # more extensive tests needed here to find bugs/missing implementation
    benchmark = FunctionApproximationBenchmark()
    benchmark.set_seed(42)
    episodes = 10

    # run
    remote_runner = RemoteRunner(benchmark, container_source=container_source)
    agent = RandomAgent(remote_runner.get_environment())
    remote_runner.run(agent, episodes)
