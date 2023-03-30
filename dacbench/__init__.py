"""DACBench: a benchmark library for Dynamic Algorithm Configuration"""
__version__ = "0.2.0"
__contact__ = "automl.org"

from dacbench.abstract_benchmark import AbstractBenchmark
from dacbench.abstract_env import AbstractEnv, AbstractMADACEnv

__all__ = ["AbstractEnv", "AbstractMADACEnv", "AbstractBenchmark"]

from gymnasium.envs.registration import register

from dacbench import benchmarks

for b in benchmarks.__all__:
    bench = getattr(benchmarks, b)()
    bench.read_instance_set()
    env_name = b[:-9]
    register(
        id=f"{env_name}-v0",
        entry_point=f"dacbench.envs:{env_name}Env",
        kwargs={"config": bench.config},
    )
