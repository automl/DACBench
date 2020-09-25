# DAClib
DAClib is a benchmark library for Dynamic Algorithm Configuration.
Its focus is on reproducibility and comparability of different DAC methods as well as easy analysis of the optimization process.

## Installation
We recommend to install DAClib in a virtual environment.
To install, run the following:
```
git clone https://github.com/automl/DAClib.git
cd DAClib
pip install .
```

## Using DAClib
Benchmarks follow the OpenAI gym standard interface. To create an environment simply:
```python
from daclib.bechmarks.sigmoid_benchmark import SigmoidBenchmark
benchmark = SigmoidBenchmark()
benchmark.config.seed = 42
env = SigmoidBenchmark.get_env()
```
The environment configuration can be changed manually or loaded from file.
Additionally, there are several wrappers with added functionality available.

## Benchmarks
Currently, DAClib includes the following Benchmarks:
- Sigmoid: tracing sigmoid curves in different dimensions and resolutions
- Luby: learning the Luby sequence

## Reproducing previous experiments
TODO
