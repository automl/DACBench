import ray
from ray import tune
from daclib.benchmarks import LubyBenchmark

# Overwrite the default config and get env
def make_luby(config):
    bench = LubyBenchmark()
    for k in config.keys():
        bench.config[k] = config[k]
    return bench.get_benchmark_env()


ray.init()
tune.register_env("luby", make_luby)
config = {"env": "luby", "env_config": {"seed": 0,}}
stop = {"training_iteration": 100}

results = tune.run("DQN", config=config, stop=stop)
ray.shutdown()
