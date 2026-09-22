.. _dacbo:

====================================
The DACBO Benchmark
====================================

| **Task:** dynamically control acquisition function parameters during Bayesian optimization
| **Cost:** symlog regret relative to a reference SMAC3 BlackBoxFacade optimizer
| **Number of hyperparameters to control:** one float (WEI α by default), or discrete acquisition function selection
| **State Information:** UBR difference, EI acquisition value, PI acquisition value, previous parameter
| **Noise Level:** moderate, depending on the BBOB target function
| **Instance space:** BBOB benchmark functions (20 functions × 3 seeds by default)

Bayesian Optimization (BO) uses a surrogate model — typically a Gaussian Process — and an
acquisition function to decide which configuration to evaluate next.
The acquisition function trades off exploration and exploitation, and its behaviour depends
heavily on one or more parameters (e.g. the exploration weight ξ for EI/PI or β for UCB).
Rather than fixing these parameters ahead of time, the DACBO benchmark frames their
adjustment as a DAC problem: at each BO trial the agent observes the current state of
the optimizer and outputs a new parameter value (or selects a different acquisition function
entirely).

Each episode runs one full BO run on a single BBOB function / seed pair drawn from the
instance set.
In every step the agent provides an action, the optimizer performs one BO trial (ask / evaluate
/ tell), and the environment returns an observation and reward.
The initial design is executed automatically during ``reset()``, so the agent only controls
the acquisition phase.
Episodes end when all trials are exhausted (``truncated``). Optionally, early termination
(``terminated``) can be enabled via ``terminate_after_reference_performance_reached=True``,
which ends an episode as soon as the incumbent surpasses the reference performance threshold.

The default reward is *symlog regret*: the difference between the current incumbent cost and
the reference performance, passed through a symmetric log transform.
This makes the reward comparable across BBOB functions whose objective values live on very
different scales.
Alternative reward signals — including raw incumbent cost, incumbent improvement, and AUC
of the optimization trajectory — can be selected via ``reward_keys``.
The default action space tunes the WEI α parameter continuously; ``AcqFunctionActionSpace``
can be used instead for discrete acquisition function selection among EI, PI, and UCB.

*The DACBO benchmark was originally developed by Carolin Benjamins as the dacboenv package
and has been integrated into DACBench. A publication is forthcoming.*

.. note::

   The DACBO environment uses numpy, scipy, and SMAC's Gaussian Process (via
   sklearn) under the hood, all of which can leverage multi-threaded BLAS
   (OpenBLAS / MKL). By default, these libraries use all available CPU cores,
   which causes oversubscription when multiple environments run in parallel
   within the same process (e.g. via ``AsyncVectorEnv`` or ``multiprocessing``).

   Set these environment variables **before importing numpy** to cap each
   process at single-threaded BLAS:

   .. code-block:: python

       import os
       os.environ["OMP_NUM_THREADS"] = "1"
       os.environ["OPENBLAS_NUM_THREADS"] = "1"
       os.environ["MKL_NUM_THREADS"] = "1"
       os.environ["NUMEXPR_NUM_THREADS"] = "1"

       import numpy  # noqa: E402

.. automodule:: dacbench.benchmarks.dacbo_benchmark
    :members:
    :show-inheritance:

.. automodule:: dacbench.envs.dacbo
    :members:
    :show-inheritance:
