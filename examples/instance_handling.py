import numpy as np
import gym

from dacbench.benchmarks import SigmoidBenchmark
from dacbench.wrappers import InstanceSamplingWrapper

def sample_sigmoid():
    rng = np.random.default_rng()
    shifts = rng.normal(5, 2.5, 1)
    slopes = (rng.choice([-1, 1], 1) * rng.uniform(size=1) * 2)
    return np.concatenate((shifts, slopes))

def sample_instance(n):
    instances = []
    for _ in range(n):
        instances.append(sample_sigmoid())
    return instances

def print_instance_set(instance_set):
    c = 1
    for i in instance_set:
        print(f"Instance {c}: {i[0]}, {i[1]}")
        c += 1

bench = SigmoidBenchmark()
bench.set_action_values([3])

instances_from_file = bench.get_benchmark_env()
print("Instance set read from file")
print_instance_set(instances_from_file.instance_set)
print("\n")

instance_set = sample_instance(20)
bench.config.instance_set = instance_set
instances_sampled_beforehand = bench.get_benchmark_env()
print("Instance set sampled before env creation")
print_instance_set(instances_sampled_beforehand.instance_set)
print("\n")

print("Instance sampled each reset")
instances_on_the_fly = InstanceSamplingWrapper(instances_from_file, sampling_function=sample_sigmoid)
print("Resetting")
instances_on_the_fly.reset()
print(f"Instance: {instances_on_the_fly.instance_set[0][0]}, {instances_on_the_fly.instance_set[0][1]}")
print("Resetting")
instances_on_the_fly.reset()
print(f"Instance: {instances_on_the_fly.instance_set[0][0]}, {instances_on_the_fly.instance_set[0][1]}")
print("Resetting")
instances_on_the_fly.reset()
print(f"Instance: {instances_on_the_fly.instance_set[0][0]}, {instances_on_the_fly.instance_set[0][1]}")
print("Resetting")
instances_on_the_fly.reset()
print(f"Instance: {instances_on_the_fly.instance_set[0][0]}, {instances_on_the_fly.instance_set[0][1]}")
print("Resetting")
print("\n")

env = SigmoidBenchmark()
print("Replacing the instance_set mid training")
env.instance_set = [[0, 0]]
print_instance_set(env.instance_set)
print("Instance set change")
env.instance_set = [[2, 1], [3, 5], [1, 1]]
print_instance_set(env.instance_set)
