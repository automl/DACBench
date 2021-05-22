.. _benchmarks:

===============
The Benchmarks
===============

.. role:: python(code)
    :language: python

DACBench currently contains six benchmarks:

* Sigmoid toy benchmark: 
  Sigmoid function approximation in multiple dimensions.
* Luby toy benchmark:
  Learning the Luby sequence.
* FastDownward:
  Heuristic selection for the FastDownward Planner.
* CMA-ES:
  Step-size adpation for CMA-ES.
* ModEA:
  Selection of Algorithm Components for EAs.
* SGD-DL:
  Learning rate adaption for a small neural network.

Our benchmarks are based on OpenAI's gym interface.
That means to run a benchmark, you need to create an environment of that benchmark
to then interact with it.
We include of this interaction between environment and DAC methods in our GitHub repository.

All of the benchmarks have a standardized version that should be used by default.
To instantiate a benchmark environment, run:

.. code-block:: python

    from dacbench.benchmarks import SigmoidBenchmark
    bench = SigmoidBenchmark()
    benchmark_env = bench.get_benchmark()

.. automodule:: dacbench.benchmarks
    :members:
    :show-inheritance:
