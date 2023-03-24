.. _contrib:

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
Adding a benchmark can be quite the project depending on the algorithms it's based on. Therefore you can always contact us via e-mail or through the issues to get assistance.

In general, there are several steps to take:

**1. Write an environment file**
This is where the functionality of your benchmark goes, in a subclass of "AbstractEnv".
Especially the "reset" and "step" functions are important as they start a run and execute the next step respectively.
Additionally, you should create a default reward function and a function to get the default observations.
We recommend using one of the simple toy environments as a model, e.g. the Sigmoid environment file.
Don't forget to enable seeding of as much of the target algorithm as possible!
You should be able to use the numpy generators provided in 'dacbench.abstract_env.AbstractEnv' the instead of 'np.random' for source of randomness and the provided seeding function in most cases.
If you need custom source of randomness e.g. for pytorch, please override the seeding function in your environment.

**2. Write a benchmark file**
This is where you specify available options for your environment, in a subclass of "AbstractBenchmark".
That includes options for the observation_space, reward range and action space.
At least one for each of these is mandatory, if you include multiple options please make sure that you also specify how to switch between them in the environment, e.g. by adding a variable for this.
Please also make sure to include a maximum number of steps per episode.
To use some specific wrappers, additional information is required. An example is the progress tracking wrapper, for which either an optimal policy for each instance or a way to compute it has to be specified.
The current benchmark classes should give you an idea how detailed these options should be.
We enourage you to provide as many possibilities to modify the benchmark as possible in order to make use of it in different scenarios, but documenting these options is important to keep the benchmark usable for others.
Again we recommend you take a look at e.g. the SigmoidBenchmark file to see how such a structure might look.

**3. Provide an instance set (or a way to sample one)**
Instances are of course important for running the benchmark.
The standard way of using them in DACBench is to read an instance set from file.
This is not mandatory, however! You can define an instance sampling method to work with our instance sampling wrapper to generate instances every episode or you can sample the instance set once before creating the environment.
How exactly you deal with instances should be specified in the benchmark class when creating the environment.
An example for the instance sampling wrapper can be found in the SigmoidBenchmark class.
Even if you provide an instance set, also making ways of sampling new instance sets possible would of course be helpful to other users.
You can furthermore provide more than one instance set, in that case it would be a good idea to label them, e.g. "wide_instance_dist" or "east_instances".

**4. Add an example use case & test cases**
To make the new benchmark accessible to everyone, please provide a small example of training an optimizer on it.
It can be short, but it should show any special cases (e.g. CMA uses a dictionary as state representation, therefore the example shows a way to flatten it into an array).
Additionally, please provide test cases for your benchmark class to ensure the environment is created properly and methods like reading the instance set work as they should.

**5. Submit a pull request**
Once everything is working, we would be grateful if you want to share the benchmark!
Submit a pull request on GitHub in which you briefly describe the benchmark and why it is interesting.
The top of the page includes some technical things to pay attention to before submitting.
Feel free to include details on how it was modelled and please cite the source if the benchmark uses existing code.

**Thank you for your contributions!**
