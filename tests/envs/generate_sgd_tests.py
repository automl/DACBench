import argparse
import os
import json
import pickle
import numpy as np

from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import SGDEnv
from dacbench.benchmarks.sgd_benchmark import SGD_DEFAULTS
from dacbench.benchmarks import SGDBenchmark
from dacbench.wrappers import ObservationWrapper

parser = argparse.ArgumentParser()
parser.add_argument('type', choices=['static', 'dynamic'])
args = parser.parse_args()

SEED = 123
np.random.seed(SEED)
benchmark = SGDBenchmark()
benchmark.config = objdict(SGD_DEFAULTS.copy())
benchmark.config.seed = SEED
benchmark.read_instance_set()
env = SGDEnv(benchmark.config)
env = ObservationWrapper(env)

data_path = os.path.join(os.path.dirname(__file__), 'data')

with open(os.path.join(data_path, 'sgd_benchmark_config.pickle'), 'wb') as f:
    pickle.dump(benchmark.config, f)

env.reset()
done = False
mem = []
step = 0

action = 0.001
while not done and step < 50:
    if args.type == 'dynamic':
        action = np.exp(np.random.uniform(-10, 1))
    state, reward, done, _ = env.step(action)

    if args.type == 'dynamic':
        params = [reward, int(done), action]
    else:
        params = [reward, int(done)]
    mem.append(np.concatenate([state, params]))
    step += 1

with open(os.path.join(data_path, f'sgd_{args.type}_test.pickle'), 'wb') as f:
    pickle.dump(np.array(mem), f)
