.. _cma:

==========================
The PyCMA CMA-ES Benchmark
==========================

| **Task:** control the step size of CMA-ES on BBOB functions
| **Cost:** negative objective value
| **Number of hyperparameters to control:** one float
| **State Information:** current point, the last 40 objective values, population size, current step size, the deltas between the last 80 objective values, the deltas between the last 40 step sizes
| **Noise Level:** fairly large, depends on target function
| **Instance space:** the BBOB functions with ids, starting point and starting sigma as well as population size

This benchmark uses the PyCMA implementation to control the step size of the CMA-ES algorithm on the BBOB function set.
The goal in the optimization is to find the global function minimum before the cutoff, so the cost is defined as the current negativ objective value.

The BBOB functions provide a varied instance space that is well suited for testing generalization capabilites of DAC methods.
Due to this large instance space and very different scales of objective values (and thus cost), the CMA-ES benchmark is one of the more difficult to solve ones in DACBench.

*The CMA-ES benchmark was constructed by Shala et al. for the paper `"Learning Step-size Adaptation in CMA-ES" <https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/20-PPSN-LTO-CMA.pdf>`_ at PPSN 2020*

.. automodule:: dacbench.benchmarks.cma_benchmark
    :members:
    :show-inheritance:

.. automodule:: dacbench.envs.cma_es
    :members:
    :show-inheritance: