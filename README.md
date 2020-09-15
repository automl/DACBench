# DAClib
A benchmark library for Dynamic Algorithm Configuration.

## Usage
Benchmarks follow the OpenAI gym standard interface. To create an environment:
```python
from daclib.bechmarks import some_benchmark
env_settings = some_benchmark.get_config(instance_set="the_instance_set_name")
env = some_benchmark.get_env(env_settings)
agent = an_agent
agent.train(env)
```
