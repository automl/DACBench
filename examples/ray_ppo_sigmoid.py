import ray
from ray import tune
from daclib.benchmarks import SigmoidBenchmark


# Overwrite standard config, but adapt action space automatically
def make_sigmoid(config):
    bench = SigmoidBenchmark()
    for k in config.keys():
        if k == "action_values":
            bench.set_action_values(config[k])
        else:
            bench.config[k] = config[k]
    return bench.get_benchmark_env()


ray.init()
tune.register_env("sigmoid", make_sigmoid)

# Play 5D scenario with irregular action count
action_values = (3, 3, 8, 8, 8)
config = {
    "env": "sigmoid",
    "env_config": {"seed": 0, "action_values": action_values,},
}
stop = {"training_iteration": 20}

results = tune.run("PPO", config=config, stop=stop)
ray.shutdown()
