.. _installation:

=======================
How to Install DACBench
=======================

.. role:: bash(code)
    :language: bash

First clone our GitHub repository:
.. code:: bash

    git clone https://github.com/automl/DACBench.git
    cd clone

We recommend installing within a virtual environment:
.. code:: bash
    conda create -n dacbench python=3.6
    conda activate dacbench

Now install DACBench with:
.. code:: bash
    pip install -e .

To also install all dependecies used in the examples, instead run:
.. code:: bash
    pip install -e .[example]