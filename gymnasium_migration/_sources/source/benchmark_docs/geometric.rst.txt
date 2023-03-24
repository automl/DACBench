.. _geometric:

===========================
The Geometric Toy Benchmark
===========================

| **Task:** aroximate values of different functions
| **Cost:** normalized distance to actual values
| **Number of hyperparameters to control:** user specified starting at one float
| **State Information:** remaining budget, derivate of each function in the last step, actual value of each function in the last step
| **Noise Level:** None
| **Instance space:** a number of different function types and their instantiations (e.g. sigmoid or linear), correlation between the functions

This is an artifical benchmark using function approximation only.
Its goal is to simulate the control of multiple hyperparameters that behave differently with possible correlations between dimensions.
In each step, the DAC controller tries to approximate the true value of the function in each dimension.
The difference between this prediction and the true value is the cost.
There are different ways to accumulate this cost built into the benchmark, by default it is the nmalized sum of costs across all dimensions.

Controlling multiple hyperparameters is a hard problem and thus this fully controllable and cheap to run benchmark aims to provide an easy starting point.
Through its flexible instance space and cost functions the difficulty can be scaled up slowly before transitioning to real-world benchmarks with multiple hyperparameters.

.. automodule:: dacbench.benchmarks.geometric_benchmark
    :members:
    :show-inheritance:

.. automodule:: dacbench.envs.geometric
    :members:
    :show-inheritance:
