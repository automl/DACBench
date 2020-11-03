# DACBench
DACBench is a benchmark library for Dynamic Algorithm Configuration.
Its focus is on reproducibility and comparability of different DAC methods as well as easy analysis of the optimization process.

If you use DACBench in you research or application, please cite us:

    @Misc{dacbench,
        author    = {T. Eimer and A. Biedenkapp and F. Hutter and M. Lindauer},
        title     = {Towards Self-Paced Context Evaluations for Contextual Reinforcement Learning},
        howpublished = {\url{https://github.com/automl/DACBench}},
        year = {2020}
    }

## Installation
We recommend to install DACBench in a virtual environment.
To install DACBench including the dependencies to run examples:
```
conda create -n dacbench python=3.6
conda activate dacbench
git clone https://github.com/automl/DACBench.git
cd DACBench
pip install -e .[example]
```
When using the Fast Downward Benchmark, you need to build it separately (we recommend cmake version 3.10.2):
```
./dacbench/envs/fast-downward/build.py
```
## Using DAClib
Benchmarks follow the OpenAI gym standard interface. That means each benchmark is created as an OpenAI gym environment. To create an environment simply:
```python
from dacbench.bechmarks.sigmoid_benchmark import SigmoidBenchmark
benchmark = SigmoidBenchmark()
benchmark.config.seed = 42
env = benchmark.get_environment()
```
The environment configuration can be changed manually or loaded from file.
Additionally, there are several wrappers with added functionality available.

## Benchmarks
Currently, DACbench includes the following Benchmarks:
- Sigmoid: tracing sigmoid curves in different dimensions and resolutions
- Luby: learning the Luby sequence
- Planning: controling the heuristics of the FastDownward Planner
- CMA-ES: adapting step-size of CMA

## Reproducing previous experiments
To reproduce the experiments from the paper a benchmark originated from, you can call
```python
from dacbench.bechmarks.sigmoid_benchmark import SigmoidBenchmark
benchmark = SigmoidBenchmark()
env = benchmark.get_benchmark(seed)
```
As some papers use different benchmark configurations, there are sometimes more options than just setting the seed.
These are:
- Sigmoid: dimension (problem dimension, in the paper either 1, 2, 3 or 5)
- Luby: L (minimum number of steps, in the paper either 8, 16 or 32) and fuzziness (in the paper 0.1, 0.8, 1.5, 2.0 and 2.5)
