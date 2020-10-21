import ray
from ray import tune
from dacbench.benchmarks import FastDownwardBenchmark


def make_fast_downward(config):
    bench = FastDownwardBenchmark()
    return bench.get_benchmark(config["seed"])


ray.init()
tune.register_env("fd", make_fast_downward)
config = {
    "env": "fd",
    "env_config": {
        "seed": 0,
    },
}
stop = {"training_iteration": 10}

results = tune.run("DQN", config=config, stop=stop)
ray.shutdown()
