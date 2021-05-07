.. _installation:

=======================
How to Install DACBench
=======================

.. role:: bash(code)
    :language: bash

First clone our GitHub repository:

.. code-block:: bash

    git clone https://github.com/automl/DACBench.git
    cd clone

We recommend installing within a virtual environment:

.. code-block:: bash

    conda create -n dacbench python=3.6
    conda activate dacbench

Now install DACBench with:

.. code-block:: bash

    pip install -e .

To also install all dependecies used in the examples, instead run:

.. code-block:: bash

    pip install -e .[example]

You should now have DACBench installed. 
In the top level directory, you will find folders for tests, examples, code coverage reporting and documentation.
The code itself can be found in the 'dacbench' folder.
If you want to take advantage of our pre-run static and random baselines (10 runs each with 1000 episodes), you can download them `here <https://www.tnt.uni-hannover.de/en/project/dacbench/>`_.
