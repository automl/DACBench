import ray
from ray import tune
from daclib.benchmarks.sigmoid_benchmark import SigmoidBenchmark

def make_sigmoid(config):
    bench = SigmoidBenchmark()
    for k in config.keys():
        bench.config[k] = config[k]
    return bench.get_benchmark_env()

ray.init()
tune.register_env("sigmoid", make_sigmoid)
config = {
    "env": "sigmoid",
    "env_config": {
        "seed": 0,
    }}
stop = {
    "training_iteration": 20
}

results = tune.run("PPO", config=config, stop=stop)
ray.shutdown()
