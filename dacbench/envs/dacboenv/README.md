This code was originally written by Carolin Benjamins and has been copied over for easier integration into DACBench. There probably will be a publication in the future on this (TODO: put it here).

## Quickstart

**Dependencies:** `smac` and `carps` must be installed (included in DACBench's optional dependencies).

```python
from dacbench.benchmarks import DACBOBenchmark

bench = DACBOBenchmark()
env = bench.get_environment()

obs, info = env.reset()

terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

Each step corresponds to one Bayesian optimization trial. The agent controls the acquisition
function parameter (WEI α by default). The initial design is run automatically during
`reset()`.

See `docs/source/benchmark_docs/dacbo.rst` for the full benchmark description, available
observation keys, reward options, and action space variants.
