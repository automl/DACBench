==============================
Functionality through Wrappers
==============================

.. role:: python(code)
    :language: python

In order to comfortably provide additional functionality to environments without changing the interface,
we can use so-called wrappers.
They execute environment resets and steps internally, but can either alter the environment behavior
(e.g. by adding noise) or record information about the environment.
To wrap an existing environment is simple:

.. code-block:: python

    from dacbench.wrappers import PerformanceTrackingWrapper

    wrapped_env = PerformanceTrackingWrapper(env)

The provided environments for tracking performance, state and action information are designed to be
used with DACBench's logging functionality.

.. automodule:: dacbench.wrappers
    :members:
