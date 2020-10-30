import ray
from ray import tune
from dacbench.benchmarks import SigmoidBenchmark


# Environment creation method
# Overwrite standard configuration with given options, but adapt action space automatically
def make_sigmoid(config):
    bench = SigmoidBenchmark()
    for k in config.keys():
        if k == "action_values":
            bench.set_action_values(config[k])
        else:
            bench.config[k] = config[k]
    return bench.get_environment()


# Initialize ray
ray.init()
# Register Sigmoid env with creation method
tune.register_env("sigmoid", make_sigmoid)

# Experiment configuration
# Play 5D scenario
action_values = (3, 3, 3, 3, 3)
sigmoid_config = {
    "env": "sigmoid",
    "env_config": {
        "seed": 0,
        "action_values": action_values,
        "instance_set_path": "../instance_sets/sigmoid_5D3M_train.csv",
    },
}
stop = {"training_iteration": 10}

# Train for 10 iterations
results = tune.run("PPO", config=sigmoid_config, stop=stop)
ray.shutdown()
