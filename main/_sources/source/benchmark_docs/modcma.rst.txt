.. _modmca:

================================
The IOHProfiler ModCMA Benchmark
================================

| **Task:** control the step size or algorithm components of CMA-ES on BBOB functions
| **Cost:** negative objective value
| **Number of hyperparameters to control:** either one float or up to 11 categoricals
| **State Information:** generation size, step size, remaining budget, function ID, instance ID
| **Noise Level:** fairly large, depends on target function
| **Instance space:** the BBOB functions with ids, starting point and starting sigma as well as population size

This benchmark is based on the IOHProfiler implementation of CMA-ES and enables both step size cool and algorithm component selection on the BBOB function set.
The components of the algorithm that can be selected or changed are: sequential execution, active update, elitism, orthogonal sampling, convergence threshold enabled,
step size adaption scheme, mirrored sampling, the base sampler, weight option, local restarts and bound correction.
The goal in the optimization is to find the global function minimum before the cutoff, so the cost is defined as the current negativ objective value.

Both versions of this benchmark are challenging due to the large instance space, but the algorithm component control adds another layer of difficulty
through its many configuration options.
It is an advanced benchmark that should likely not be the starting point for the development of DAC methods.

.. automodule:: dacbench.benchmarks.modcma_benchmark
    :members:
    :show-inheritance:

.. automodule:: dacbench.envs.modcma
    :members:
    :show-inheritance: