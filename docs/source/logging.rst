===================
Logging Experiments
===================

.. role:: python(code)
    :language: python

As there are many potentially interesting metrics involved in the analysis of DAC methods,
DACBench includes the possibility to track and store them.

To log information on an environment, you need a logger object:

.. code-block:: python

    from dacbench.logger import Logger
    from pathlib import Path

    logger = Logger(experiment_name="example", output_path=Path("your/path"))

If you want to use any of our tracking wrappers, you can then create a logging module for them:

.. code-block:: python

    from dacbench.wrappers import PerformanceTrackingWrapper

    performance_logger = logger.add_module(PerformanceTrackingWrapper)
    env = PerformanceTrackingWrapper(env, logger=performance_logger)
    logger.add_env()

Now the logger will store information in the specified directory in .jsonl files.
By adding more wrappers, you will also be provided with more information.
The stored data can then be loaded into pandas dataframes:

.. code-block:: python

    from dacbench.logger import load_logs, log2dataframe

    logs = load_logs("your/path/PerformancyTrackingWrapper.jsonl")
    df = log2dataframe(logs)

.. automodule:: dacbench.logger
    :members:
