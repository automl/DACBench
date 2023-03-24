=========================================
Saving & Loading Benchmark Configurations
=========================================

.. role:: python(code)
    :language: python

While we encourage the use of the default benchmark settings,
we recognize that our benchmarks are not perfect and can be improved upon.
Therefore, it is possible to modify benchmarks and save these modifications to share with others.

To load a configuration shared with you, read it using the corresponding benchmark class:

.. code-block:: python

    from dacbench.benchmarks import SigmoidBenchmark

    bench = SigmoidBenchmark()
    bench.read_config_file("path/to/your/config.json")
    modified_benchmark = bench.get_environment()

The get_environment() method overrides wth default configurations with your changes.
That way you can directly modify the benchmarks:

.. code-block:: python

    from dacbench.benchmarks import SigmoidBenchmark

    bench = SigmoidBenchmark()

    # Increase episode length
    bench.config.cutoff = 20
    # Decrease slope multiplier
    bench.config.slope_multiplier = 1.5

    modified_benchmark = bench.get_environment()

To then save this configuration:

.. code-block:: python

    bench.save_config("your/path/config.json")

In case you want to modify state information, reward function or other complex benchmark attributes,
be sure to adapt all dependencies in the configuration.
Benchmarks have methods to do this for common changes like the number of dimensions in Sigmoid.

If any of your changes pass a function to the configuration,
please be sure to provide the code for this function along with the configuration itself.
If you want to save wrappers to the config (e.g. an instance sampling wrapper),
you need to register them beforehand and also provide any functions that may serve as arguments.
