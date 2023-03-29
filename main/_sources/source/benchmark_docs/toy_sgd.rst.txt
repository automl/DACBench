.. _toysgd:

====================
The ToySGD Benchmark
====================

| **Task:** control the learning rate and momentum of SGD in simple function approximation
| **Cost:** log regret
| **Number of hyperparameters to control:** two floats
| **State Information:** remaining budget, gradient, current learning rate, current momentum
| **Noise Level:** fairly small
| **Instance space:** target function specification

This artificial benchmark uses functions like polynomials to test DAC controllers' ability to control both learning rate and momentum of SGD.
At each step until the cutoff, both hyperparameters are updated and one optimization step is taken.
As we know the global optimum of the function, the cost is measured as the current regret.

By using function approximation, this benchmark is computationally cheap, so likely a good entry point before tackling the full-sizes SGD or CMA-ES step size benchmarks.
It can also serve as a first test whether a DAC method can handle multiple hyperparameters at the same time.

.. automodule:: dacbench.benchmarks.toysgd_benchmark
    :members:
    :show-inheritance:

.. automodule:: dacbench.envs.toysgd
    :members:
    :show-inheritance: