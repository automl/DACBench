.. _benchmarks:

===============
The Benchmarks
===============

.. role:: python(code)
    :language: python

DACBench contains a range of benchmarks in different categories and from different domains.
There is a range of highly configurable, cheap to run benchmarks that often also include a
ground truth solution.
We recommend using these as an introduction to DAC, to verify new algorithms and to
generate detailed insights.
They are both based on artificial functions and real algorithms:

- :doc:`Sigmoid </benchmark_docs/sigmoid>`_ (Artificial Benchmark):
  Sigmoid function approximation in multiple dimensions.
- :doc:`Luby </benchmark_docs/luby>`_ (Artificial Benchmark):
  Learning the Luby sequence.
- ToySGD (Artificial Benchmark):
  Controlling the learning rate in gradient descent.
- :doc:`Geometric </benchmark_docs/geometric>`_ (Artificial Benchmark):
  Approximating several functions at once.
- Toy version of the :doc:`FastDownward benchmark </benchmark_docs/fastdownward>`_:
  Heuristic selection for the FastDownward Planner with ground truth.
- Theory benchmark with ground truth:
  RLS algorithm on the LeadingOnes problem.


Beyond these smaller scale problems we know a lot about, DACBench also contains less
interpretable algorithms with larger scopes. These are oftentimes noisier, harder to debug
and more costly to run and thus present a real challenge for DAC algorithms:

- :doc:`FastDownward benchmark </benchmark_docs/fastdownward>`_:
  Heuristic selection for the FastDownward Planner on competition tasks.
- :doc:`CMA-ES </benchmark_docs/cma>`_:
  Step-size adpation for CMA-ES.
- ModEA:
  Selection of Algorithm Components for EAs.
- :doc:`ModCMA </benchmark_docs/modcma>`_:
  Step-size & algorithm component control for EAs backed by IOHProfiler.
- SGD-DL:
  Learning rate adaption for neural networks.

Our benchmarks are based on OpenAI's gym interface for Reinforcement Learning.
That means to run a benchmark, you need to create an environment of that benchmark
to then interact with it.
We include examples of this interaction between environment and DAC methods in our
GitHub repository.
To instantiate a benchmark environment, run:

.. code-block:: python

    from dacbench.benchmarks import SigmoidBenchmark
    bench = SigmoidBenchmark()
    benchmark_env = bench.get_environment()

.. autoclass:: dacbench.abstract_benchmark
    :members:
    :show-inheritance:

.. automodule:: dacbench.abstract_env
    :members:
    :show-inheritance: