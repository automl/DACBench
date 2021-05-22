.. _benchmarks:

========================
Contributing to DACBench
========================

.. role:: bash(code)
    :language: bash

DACBench is an open-source collaborative project. 
Since its conception, we have had several valueable contributions and appreciate everyone who wants to make DACBench bigger and better for the community.
This document can be a guide on how to get started contributing to DACBench.

In general, there are many ways you can help improve DACBench, including:

* Contributing new benchmarks
* Extending existing benchmarks (by adding e.g. more hyperparameters, extending the state information or providing interesting instances)
* Maining the code & fixing bugs
* Improving the documentation

For most of these, the existing issues should give you an idea where to start. 
If something is missing or not working for you, consider opening an issue on it, especially if it can't be fixed within a few minutes.
Issues are also a good place to request new features and extensions, so don't hesitate to create one.

Guidelines for Pull-Requests
############################
Code contributions are best made through pull-requests. 
In order to make the integration as easy as possible, we ask that you follow a few steps:

1. Please describe the changes you made in the PR clearly. This makes reviewing much faster and avoids misunderstandings
2. Run our tests and ideally also test coverage before submitting so your PR doesn't accidentally introduce new errors. You can use pytest for both of these, to only test, run from the top level DACBench dir:

.. code-block:: bash

        pytest tests

For tests and test coverage:

.. code-block:: bash

        pytest --cov=dacbench --cov-report html tests

3. If you install the 'dev' extras of DACBench, you should have flake8 and the code formatting tool black setup in a pre-commit hook. Both ensure consistent code quality, so ensure that the format is correct.
4. If you make larger changes to the docs, please build them locally using Sphinx. If you're not familiar with the tool, you can find a guide here: https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html

Adding a Benchmark
############################
If you're adding or extending a benchmark, please also use the numpy generators provided in 'dacbench.abstract_env.AbstractEnv' the instead of 'np.random' for source of randomness and use the provided seeding function as well. If you need custom source of randomness e.g. for pytorch, please override the seeding function in your environment.

Thank you for your contributions!
