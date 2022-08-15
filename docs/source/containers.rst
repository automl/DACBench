.. _containers:

=========================
Using DACBench Containers
=========================

.. role:: bash(code)
    :language: bash

DACBench can run containerized versions of Benchmarks using Singularity containers to
isolate their dependencies and make reproducible Singularity images.

To build an existing container, install Singularity and run the following to build the
container of your choice. Here is an example for the CMA container:

.. code-block:: bash

    cd dacbench/container/singularity_recipes
    sudo singularity build cma cma.def

An example on how to use the container can be found in the examples in the repository.

For writing your own recipe to build a Container, you can refer to the recipe template in the
repository: dacbench/container/singularity_recipes/recipe_template