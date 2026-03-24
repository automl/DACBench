# flake8: noqa: F401
import importlib
import warnings

# from dacbench.envs.fast_downward import FastDownwardEnv
from dacbench.envs.function_approximation import (
    FunctionApproximationEnv,
    FunctionApproximationInstance,
)
from dacbench.envs.luby import LubyEnv, LubyInstance, luby_gen
from dacbench.envs.theory import TheoryEnv
from dacbench.envs.toysgd import ToySGDEnv, ToySGDInstance

__all__ = [
    "FunctionApproximationEnv",
    "FunctionApproximationInstance",
    "LubyEnv",
    "LubyInstance",
    "TheoryEnv",
    #   "FastDownwardEnv",
    "ToySGDEnv",
    "ToySGDInstance",
    "luby_gen",
]

modcma_spec = importlib.util.find_spec("modcma")
found = modcma_spec is not None
if found:
    from dacbench.envs.cma_es import CMAESEnv, CMAESInstance

    __all__.append("CMAESEnv")
    __all__.append("CMAESInstance")
else:
    warnings.warn(  # noqa: B028
        "CMA-ES Benchmark not installed. If you want to use this benchmark, "
        "please follow the installation guide."
    )

sgd_spec = importlib.util.find_spec("torch")
found = sgd_spec is not None
if found:
    from dacbench.envs.sgd import SGDEnv, SGDInstance

    __all__.append("SGDEnv")
    __all__.append("SGDInstance")
else:
    warnings.warn(  # noqa: B028
        "SGD Benchmark not installed. If you want to use this benchmark, "
        "please follow the installation guide."
    )

dacbo_deps_spec = importlib.util.find_spec("smac") and importlib.util.find_spec("carps")
found = dacbo_deps_spec is not None
if found:
    from dacbench.envs.dacbo import DACBOEnv

    __all__.append("DACBOEnv")
else:
    warnings.warn(  # noqa: B028
        "DACBO Env dependencies (smac, carps) not installed. "
        "If you want to use this benchmark, "
        "please follow the installation guide."
    )
