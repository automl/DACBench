"""
This script can be used to generate test cases for SGD benchmark
using the current default config.
"""
import os
import numpy as np
import json
import hashlib

from dacbench.abstract_benchmark import objdict
from dacbench.envs import SGDEnv
from dacbench.benchmarks.sgd_benchmark import SGD_DEFAULTS
from dacbench.benchmarks import SGDBenchmark
from dacbench.wrappers import ObservationWrapper

types = ['static', 'dynamic']

SEED = 123
data_path = os.path.join(os.path.dirname(__file__), 'data')

string_config = {k: str(v) for (k, v) in SGD_DEFAULTS.items()}
h = hashlib.sha1(json.dumps(string_config).encode()).hexdigest()

def get_environment():
    benchmark = SGDBenchmark()
    env = benchmark.get_benchmark(seed=SEED)
    env = ObservationWrapper(env)
    return env

with open(os.path.join(data_path, f'sgd_config.hash'), 'w') as f:
    f.write(h)

for type in types:
    env = get_environment()
    env.reset()
    done = False
    mem = []
    step = 0
    action = 0.001
    while not done and step < 50:
        if type == 'dynamic':
            action = np.exp(np.random.uniform(-10, 1))
        state, reward, done, _ = env.step(action)
        mem.append(np.concatenate([state, [reward, int(done), action]]))
        step += 1

    np.save(os.path.join(data_path, f'sgd_{type}_test'), np.array(mem))
