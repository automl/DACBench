.. _theory:

=====================
The Theory Benchmark
=====================

| **Task:** controlling number of flips in RLS on LeadingOnes
| **Cost:** number of iterations until solution
| **Number of hyperparameters to control:** one float
| **State Information:** user specified, highly flexible
| **Noise Level:** fairly large
| **Instance space:** different instantiations of LeadingOnes

This benchmark is considered one of our highly controllable ones as there is ground truth available.
It is also, however, built on top of the RLS algorithm, so not an artificial benchmark.
At each step, the DAC controller chooses how many solution bits to flip.
We want to optimize how many algorithm steps are taken, so the number of iterations is the reward.

While this is not an easy to solve benchmark, it is cheap to run and interfaces a real EA.
Thus it may be a good entry point for harder EA-based benchmarks and also a good benchmark for analyzing controller behavior.

*The Theory benchmark was constructed by Biedenkapp et al. for the paper `"Theory-Inspired Parameter Control Benchmarks for Dynamic Algorithm Configuration" <https://arxiv.org/pdf/2202.03259.pdf>`_ at GECCO 2022*


.. automodule:: dacbench.benchmarks.theory_benchmark
    :members:
    :show-inheritance:

.. automodule:: dacbench.envs.theory
    :members:
    :show-inheritance: