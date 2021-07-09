"""
This script can be used to generate test cases for SGD benchmark.
Config file is saved seperately and then it is used to generate test cases so that,
config file can be regenerated after a refactoring without touching the test case files.
"""
import argparse
import os
import pickle
import numpy as np

from dacbench.abstract_benchmark import objdict
from dacbench.envs import SGDEnv
from dacbench.benchmarks.sgd_benchmark import SGD_DEFAULTS
from dacbench.benchmarks import SGDBenchmark
from dacbench.wrappers import ObservationWrapper

parser = argparse.ArgumentParser()
parser.add_argument('type', choices=['config', 'static', 'dynamic'])
args = parser.parse_args()

benchmark = SGDBenchmark()
SEED = 123
data_path = os.path.join(os.path.dirname(__file__), 'data')

if args.type == 'config':
    with open(os.path.join(data_path, 'sgd_benchmark_config.pickle'), 'wb') as f:
        np.random.seed(SEED)
        benchmark.config = objdict(SGD_DEFAULTS.copy())
        benchmark.config.seed = SEED
        benchmark.read_instance_set()
        pickle.dump(benchmark.config, f)
        exit(0)

config_path = os.path.join(data_path, 'sgd_benchmark_config.pickle')
if not os.path.exists(config_path):
    raise FileNotFoundError('''"sgd_benchmark_config.pickle" is not found.\
 Call "python generate_sgd_tests.py config" to generate it.''')

with open(config_path, 'rb') as f:
    benchmark.config = pickle.load(f)

env = SGDEnv(benchmark.config)
env = ObservationWrapper(env)

env.reset()
done = False
mem = []
step = 0

action = 0.001
while not done and step < 50:
    if args.type == 'dynamic':
        action = np.exp(np.random.uniform(-10, 1))
    state, reward, done, _ = env.step(action)
    mem.append(np.concatenate([state, [reward, int(done), action]]))
    step += 1

with open(os.path.join(data_path, f'sgd_{args.type}_test.pickle'), 'wb') as f:
    pickle.dump(np.array(mem), f)
