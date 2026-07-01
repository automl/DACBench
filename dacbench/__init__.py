"""DACBench: a benchmark library for Dynamic Algorithm Configuration."""

try:
    from dacbench._version import version as __version__
except ImportError:
    # Package not built with setuptools_scm (e.g. raw source tree, no install).
    # Fall back to installed metadata, then a hardcoded tail value.
    try:
        from importlib.metadata import (
            PackageNotFoundError,
            version as _pkg_version,
        )

        __version__ = _pkg_version("DACBench")
    except PackageNotFoundError:
        __version__ = "0.4.0.dev0"

__contact__ = "automl.org"

from dacbench.abstract_benchmark import AbstractBenchmark
from dacbench.abstract_env import AbstractEnv, AbstractMADACEnv

__all__ = ["AbstractBenchmark", "AbstractEnv", "AbstractMADACEnv"]

from gymnasium.envs.registration import register

from dacbench import benchmarks

try:
    for b in benchmarks.__all__:
        if b == "FastDownwardBenchmark":
            continue
        bench = getattr(benchmarks, b)()
        bench.read_instance_set()
        env_name = b[:-9]
        register(
            id=f"{env_name}-v0",
            entry_point=f"dacbench.envs:{env_name}Env",
            kwargs={"config": bench.config},
        )
except:  # noqa: E722
    print(
        "DACBench Gym registration failed - make sure you have all dependencies "
        "installed and their instance sets in the right path!"
    )
