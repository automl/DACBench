.. _modea:

===================
The ModEA Benchmark
===================

**Task:** control the algorithm components of CMA-ES on BBOB functions
**Cost:** negative objective value
**Number of hyperparameters to control:** 11 categorical
**State Information:** generation size, step size, remaining budget, function ID, instance ID
**Noise Level:** fairly large, depends on target function
**Instance space:** the BBOB functions with ids, starting point and starting sigma as well as population size

This benchmark uses the ModEA package to enable dynamic control of several algorithm components of CMA-ES.
The components of the algorithm that can be selected or changed are: sequential execution, active update, elitism, orthogonal sampling, convergence threshold enabled,
step size adaption scheme, mirrored sampling, the base sampler, weight option, local restarts and bound correction.
The goal in the optimization is to find the global function minimum before the cutoff, so the cost is defined as the current negativ objective value.

Just like the ModCMA benchmark (which provides a very similar problem with a different backend), this benchmark is challenging due to the large configuration space.
It is an advanced benchmark that should likely not be the starting point for the development of DAC methods.

.. automodule:: dacbench.benchmarks.modea_benchmark
    :members:
    :show-inheritance:

.. automodule:: dacbench.envs.modea
    :members:
    :show-inheritance: