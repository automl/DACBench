.. _sgd:

===============================
The SGD Deep Learning Benchmark
===============================

| **Task:** control the learning rate in deep learning
| **Cost:** log differential validation loss
| **Number of hyperparameters to control:** one float
| **State Information:** predictive change variance, predictive change variance, loss variance, loss variance uncertainty, current learning rate, training loss, validation loss, step, alignment, crashed
| **Noise Level:** fairly large
| **Instance space:** dataset, network architecture, optimizer

Built on top of PyTorch, this benchmark allows for dynamic learning rate control in deep learning.
At each step until the cutoff, i.e. after each epoch, the DAC controller provides a new learning rate value to the network.
Success is measured by decreasing validation loss.

This is a very flexible benchmark, as in principle all kinds of classification datasets and PyTorch compatible architectures can be included in training.
The underlying task is not easy, however, so we recommend starting with small networks and datasets and building up to harder tasks.

.. automodule:: dacbench.benchmarks.sgd_benchmark
    :members:
    :show-inheritance:

.. automodule:: dacbench.envs.sgd
    :members:
    :show-inheritance: