import ray
from ray import tune
from dacbench.benchmarks import FastDownwardBenchmark


# Method to create env
# Here we use the published version of the FastDownward Benchmark
def make_fast_downward(config):
    bench = FastDownwardBenchmark()
    return bench.get_benchmark(config["seed"])


# Initialize ray
ray.init()
# Register env with creation method
tune.register_env("fd", make_fast_downward)

# Experiment configuration
config = {"env": "fd", "env_config": {"seed": 0}}
stop = {"training_iteration": 10}

# Run DQN
results = tune.run("DQN", config=config, stop=stop)
ray.shutdown()
