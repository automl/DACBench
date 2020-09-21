from daclib.benchmarks import Benchmark
from daclib.wrappers import RewardNoiseWrapper

def train_agent(agent, config_path):
    #Load config from file, but overwrite with custom settings
    benchmark = Benchmark(config_path, seed=0)
    config = Benchmark.get_config()

    #Generate env with specified attributes
    env = Benchmark.get_benchmark_env()

    #Wrap env for more information
    env = RewardNoiseWrapper(env, config)

    #Now you have a normal gym env to train on
    agent.train(env)
