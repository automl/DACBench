.. _luby:

======================
The Luby Toy Benchmark
======================

| **Task:** lning the Luby sequence with variations
| **Cost:** correctness of sequence element prediction
| **Number of hyperparameters to control:** one integer
| **State Information:** Actions and timesteps of the last three iterations
| **Noise Level:** None
| **Instance space:** the Luby sequence with possibilities to modify the starting point of the series (e.g. element 5 instead of 1) as well as the repetition fo each element

This benchmark is not built on top of an algorithm, instead it's a pure sequence learning task.
In each step until the cutoff, the DAC controller's task is to predict the next element of the Luby sequence.
If the prediction is correct, it is given a reward of 1 and else 0.

The benchmark is very cheap to run, but can be altered to be quite challenging nonetheless.
In its basic form, it can serve to validate DAC methods and observe their prowess in learning a series of predictions correctly.

*The Luby benchmark was constructed by Biedenkapp et al. for the paper `"Dynamic Algorithm Configuration: Foundation of a New Meta-Algorithmic Framework" <https://www.tnt.uni-hannover.de/papers/data/1432/20-ECAI-DAC.pdf>`_ at ECAI 2020*

.. automodule:: dacbench.benchmarks.luby_benchmark
    :members:
    :show-inheritance:

.. automodule:: dacbench.envs.luby
    :members:
    :show-inheritance: