# flake8: noqa: F401
import importlib
import warnings

from dacbench.benchmarks.fast_downward_benchmark import FastDownwardBenchmark
from dacbench.benchmarks.geometric_benchmark import GeometricBenchmark
from dacbench.benchmarks.luby_benchmark import LubyBenchmark
from dacbench.benchmarks.sigmoid_benchmark import SigmoidBenchmark
from dacbench.benchmarks.toysgd_benchmark import ToySGDBenchmark

__all__ = [
    "LubyBenchmark",
    "SigmoidBenchmark",
    "ToySGDBenchmark",
    "GeometricBenchmark",
    "FastDownwardBenchmark",
]


cma_spec = importlib.util.find_spec("cma")
found = cma_spec is not None
if found:
    from dacbench.benchmarks.cma_benchmark import CMAESBenchmark

    __all__.append("CMAESBenchmark")
else:
    warnings.warn(
        "CMA-ES Benchmark not installed. If you want to use this benchmark, please follow the installation guide."
    )

modcma_spec = importlib.util.find_spec("modcma")
found = modcma_spec is not None
if found:
    from dacbench.benchmarks.modcma_benchmark import ModCMABenchmark

    __all__.append("ModCMABenchmark")
else:
    warnings.warn(
        "ModCMA Benchmark not installed. If you want to use this benchmark, please follow the installation guide."
    )

sgd_spec = importlib.util.find_spec("backpack")
found = sgd_spec is not None
if found:
    from dacbench.benchmarks.sgd_benchmark import SGDBenchmark

    __all__.append("SGDBenchmark")
else:
    warnings.warn(
        "SGD Benchmark not installed. If you want to use this benchmark, please follow the installation guide."
    )

theory_spec = importlib.util.find_spec("uuid")
found = theory_spec is not None
if found:
    from dacbench.benchmarks.theory_benchmark import TheoryBenchmark

    __all__.append("TheoryBenchmark")
else:
    warnings.warn(
        "Theory Benchmark not installed. If you want to use this benchmark, please follow the installation guide."
    )
