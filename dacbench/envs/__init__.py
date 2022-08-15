# flake8: noqa: F401
from dacbench.envs.luby import LubyEnv, luby_gen
from dacbench.envs.sigmoid import (
    SigmoidEnv,
    ContinuousSigmoidEnv,
    ContinuousStateSigmoidEnv,
)
from dacbench.envs.fast_downward import FastDownwardEnv
from dacbench.envs.toysgd import ToySGDEnv
from dacbench.envs.geometric import GeometricEnv

__all__ = ["LubyEnv", "luby_gen", "SigmoidEnv", "FastDownwardEnv", "ToySGDEnv", "GeometricEnv"]


import importlib
import warnings

cma_spec = importlib.util.find_spec("cma")
found = cma_spec is not None
if found:
    from dacbench.envs.cma_es import CMAESEnv

    __all__.append("CMAESEnv")
else:
    warnings.warn(
        "CMA-ES Benchmark not installed. If you want to use this benchmark, please follow the installation guide."
    )

modea_spec = importlib.util.find_spec("modea")
found = modea_spec is not None
if found:
    from dacbench.envs.modea import ModeaEnv

    __all__.append("ModeaEnv")
else:
    warnings.warn(
        "Modea Benchmark not installed. If you want to use this benchmark, please follow the installation guide."
    )

modcma_spec = importlib.util.find_spec("modcma")
found = modcma_spec is not None
if found:
    from dacbench.envs.cma_step_size import CMAStepSizeEnv
    from dacbench.envs.modcma import ModCMAEnv

    __all__.append("ModCMAEnv")
    __all__.append("CMAStepSizeEnv")
else:
    warnings.warn(
        "ModCMA Benchmark not installed. If you want to use this benchmark, please follow the installation guide."
    )

sgd_spec = importlib.util.find_spec("torchvision")
found = sgd_spec is not None
if found:
    from dacbench.envs.sgd import SGDEnv

    __all__.append("SGDEnv")
else:
    warnings.warn(
        "SGD Benchmark not installed. If you want to use this benchmark, please follow the installation guide."
    )

theory_spec = importlib.util.find_spec("uuid")
found = theory_spec is not None
if found:
    from dacbench.envs.theory import RLSEnvDiscrete, RLSEnv

    __all__.append("RLSEnv")
    __all__.append("RLSEnvDiscrete")
else:
    warnings.warn(
        "Theory Benchmark not installed. If you want to use this benchmark, please follow the installation guide."
    )
