# Repo Structure

This file gives on overview the poruses of the different files and folders: 

## Repo Level
```
├── dacbench:               python package
├── docs:                   configs and style descriptions for the docs. To build the docs it is required to install the docs extra
├── examples:               example scrips/usages for different parts of the library. Requires the example extra
├── Getting started.ipynb:  a jupyter notebook showing basic usage of DACBench
├── LICENSE:                the licence for this software
├── pyproject.toml:         configuration file for setuptools, tests (pytest, coverage), black (code formatter)
├── README.md:              project readme
├── runscripts:             shell scripts to run baselines on  SLURM
├── setup.cfg:              pip-package config. Contains: dependencies, extras and thier dependencies, tags, project meta
├── setup.py:               dummy setup.py require for downward compatibility          
├── tests:                  unit tests for dacbench
└── train_ppo.py
``
