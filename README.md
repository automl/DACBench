
# DACBench
[![Documentation Status](https://readthedocs.org/projects/dacbench/badge/?version=latest)](https://dacbench.readthedocs.io/en/latest/?badge=latest)
[![Run Python Tests](https://github.com/automl/DACBench/actions/workflows/run-python-tests.yml/badge.svg)](https://github.com/automl/DACBench/actions/workflows/run-python-tests.yml)

DACBench is a benchmark library for Dynamic Algorithm Configuration.
Its focus is on reproducibility and comparability of different DAC methods as well as easy analysis of the optimization process.

If you use DACBench in your research or application, please cite us:

```bibtex
@inproceedings{eimer-ijcai21,
  author    = {T. Eimer and A. Biedenkapp and M. Reimer and S. Adriaensen and F. Hutter and M. Lindauer},
  title     = {DACBench: A Benchmark Library for Dynamic Algorithm Configuration},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence ({IJCAI}'21)},
  year      = {2021},
  month     = aug,
  publisher = {ijcai.org},

## Getting Started

### Installation
We recommend installing DACBench in a virtual environment:
```
conda create -n dacbench python=3.6
conda activate dacbench
git clone https://github.com/automl/DACBench.git
cd DACBench
git submodule update --init --recursive
pip install .
```
This command installs the base version of DACBench including the three small surrogate benchmarks and the option to install the FastDownward benchmark.
For any other benchmark, you may use a singularity container as provided by us (see next section) or install it as an additional dependency. As an example, 
to install the SGDBenchmark, run:

```
pip install -e .[sgd]
```

To use FastDownward, you should first install the solver itself.

### Running a benchmark
Our 'examples' directory contains several uses of our benchmarks with RL agents. Next to our 'Getting Started' jupyter notebook, it's a good resource to
understand the interface of DACBench. In order to run the examples, please be sure to install the example dependencies:
```
pip install -e .[example]
```

## Containerized Benchmarks

DACBench can run containerized versions of Benchmarks using Singularity containers to isolate their dependencies and make reproducible Singularity images. 


### Building a Container

For writing your own recipe to build a Container, you can refer to `dacbench/container/singularity_recipes/recipe_template`  

Install Singularity and run the following to build the (in this case) cma container

```bash
cd dacbench/container/singularity_recipes
sudo singularity build cma cma.def
```

## More DAC
