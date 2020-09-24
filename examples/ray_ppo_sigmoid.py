import ray
from ray import tune
import numpy as np
from daclib.benchmarks.sigmoid_benchmark import SigmoidBenchmark

def make_sigmoid(config):
    bench = SigmoidBenchmark()
    for k in config.keys():
        bench.config[k] = config[k]
    return bench.get_benchmark_env()

ray.init()
tune.register_env("sigmoid", make_sigmoid)

#Play 5D scenario
action_values = (3, 3, 3, 3, 3)
config = {
    "env": "sigmoid",
    "env_config": {
        "seed": 0,
        "action_values": action_values,
        "observation_space_args": [np.array([-np.inf for _ in range(1 + len(action_values) * 3)]), np.array([np.inf for _ in range(1 + len(action_values) * 3)])],
        "action_space_args": [int(np.prod(action_values))]
    }}
stop = {
    "training_iteration": 20
}

results = tune.run("PPO", config=config, stop=stop)
ray.shutdown()
