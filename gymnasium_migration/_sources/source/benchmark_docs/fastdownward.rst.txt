.. _fastdownward:

==========================
The FastDownward Benchmark
==========================

| **Task:** select heuristics for the FastDownward planner
| **Cost:** number of optimization steps
| **Number of hyperparameters to control:** one categorical
| **State Information:** average value, max value, min value, number of open list entries and variance for each heuristic
| **Noise Level:** fairly large
| **Instance space:** either specifically desigd easy toy instances with ground truth or common planning competition instance sets

This benchmark is an interface to the Fast Downward AI planner, controlling its heuristic hyperparameter.
In each step until the algorithm finishes or is terminated via the cutoff, the DAC controller selects one of either two
(toy case) or four heuristiccs for the planner to use.
The goal is to reduce the runtime of the planner, so every step that is taken in the benchmark incurs a cost of 1.

Out of our real-world benchmarks, FastDownward is likely the fastest running and it has been shown to be suitable to dynamic configuration.
Though the noise level is fairly high, most DAC controllers should be able to learn functional policies in a comparatively short time frame.

*The FastDownward benchmark was constructed by Speck et al. for the paper `"Learning Heuristic Selection with Dynamic Algorithm Configuration" <https://arxiv.org/pdf/2006.08246.pdf>`_ at ICAPS 2021*


.. automodule:: dacbench.benchmarks.fast_downward_benchmark
    :members:
    :show-inheritance:

.. automodule:: dacbench.envs.fast_downward
    :members:
    :show-inheritance: