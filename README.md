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
When using the Fast Downward Benchmark, you need to build it separately:
```
./daclib/envs/fast-downward/build.py
```
## Using DAClib
Benchmarks follow the OpenAI gym standard interface. To create an environment simply:
```python
from daclib.bechmarks.sigmoid_benchmark import SigmoidBenchmark
benchmark = SigmoidBenchmark()
benchmark.config.seed = 42
env = SigmoidBenchmark.get_benchmark_env()
```
The environment configuration can be changed manually or loaded from file.
Additionally, there are several wrappers with added functionality available.

## Benchmarks
Currently, DAClib includes the following Benchmarks:
- Sigmoid: tracing sigmoid curves in different dimensions and resolutions
- Luby: learning the Luby sequence

## Reproducing previous experiments
To reproduce the experiments from the paper a benchmark originated from, you can call
```python
from daclib.bechmarks.sigmoid_benchmark import SigmoidBenchmark
benchmark = SigmoidBenchmark()
env = SigmoidBenchmark.get_benchmark(seed)
```
As some papers use different benchmark configurations, there are sometimes more options than just setting the seed.
These are:
- Sigmoid: dimension (problem dimension, in the paper either 1, 2, 3 or 5)
- Luby: L (minimum number of steps, in the paper either 8, 16 or 32) and fuzziness (in the paper 0.1, 0.8, 1.5, 2.0 and 2.5)
- Fast Downward:
- CMA:
