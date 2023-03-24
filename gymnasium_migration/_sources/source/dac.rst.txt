.. _dac:

==================================================
Dynamic Algorithm Configuration - A Short Overview
==================================================

**Dynamic Algorithm Configuration (DAC)** [`Biedenkapp et al., ECAI 2020 <https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/20-ECAI-DAC.pdf>`_, `Adriaensen et al., CoRR 2022 <https://arxiv.org/pdf/2205.13881.pdf>`_] is a
paradigm for hyperparameter optimization that aims to find the best possible configuration
of algorithm hyperparameters for *each step* in the algorithm's execution and for *each algorithm
instance*.

That means DAC methods configure hyperparameters dynamically over the runtime of an algorithm
because the optimal value could be very different at the start than in the end. An example for
this is the learning rate in SGD where at first we want to traverse the loss landscape fairly
quickly (= high learning rate), but then need to slow down as to not overshoot the optimum
(= gradually decreasing learning rate).

Furthermore, as we configure across a set of instances, DAC methods also need to take into
account how these factors change between algorithm instances - a learning rate schedule on a
very simple image classification problem like MNIST, for example, will likely look different
than on a challenging one like ImageNet.

DAC can be solved by a number of different methods, e.g. classical Algorithm Configuration
tools or Reinforcement Learning. As the paradigm is still relatively new, there is a lot of
space to experiment with new possibilities or combine existing configuration options.
In order to stay up to date on the current DAC literatur, we recommend our *`DAC literature
overview <https://www.automl.org/automated-algorithm-design/dac/literature-overview/>`_*.