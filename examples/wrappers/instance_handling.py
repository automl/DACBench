"""Example for instance handling."""
import numpy as np
from dacbench.benchmarks import SigmoidBenchmark
from dacbench.wrappers import InstanceSamplingWrapper


# Helper method to sample a single sigmoid instance
def sample_sigmoid():
    rng = np.random.default_rng()
    shifts = rng.normal(5, 2.5, 1)
    slopes = rng.choice([-1, 1], 1) * rng.uniform(size=1) * 2
    return np.concatenate((shifts, slopes))


# Sample n sigmoid instances
def sample_instance(n):
    instances = {}
    for _ in range(n):
        instances[n] = sample_sigmoid()
    return instances


# Helper method to print current instance set
def print_instance_set(instance_set):
    c = 1
    for i in instance_set:
        print(f"Instance {c}: {instance_set[i][0]}, {instance_set[i][1]}")
        c += 1


# Make Sigmoid benchmark object
bench = SigmoidBenchmark()
bench.set_action_values([3])

# First example: read instances from default instance set path
instances_from_file = bench.get_environment()
print("Instance set read from file")
print_instance_set(instances_from_file.instance_set)
print("\n")

# Second example: Sample instance set before training
instance_set = sample_instance(20)
bench.config.instance_set = instance_set
instances_sampled_beforehand = bench.get_environment()
print("Instance set sampled before env creation")
print_instance_set(instances_sampled_beforehand.instance_set)
print("\n")

# Third example: Sample instances during training using the InstanceSamplingWrapper
print("Instance sampled each reset")
instances_on_the_fly = InstanceSamplingWrapper(
    instances_from_file, sampling_function=sample_sigmoid
)
print("Resetting")
instances_on_the_fly.reset()
print(
    f"Instance: {instances_on_the_fly.instance_set[0][0]}, {instances_on_the_fly.instance_set[0][1]}"
)
print("Resetting")
instances_on_the_fly.reset()
print(
    f"Instance: {instances_on_the_fly.instance_set[0][0]}, {instances_on_the_fly.instance_set[0][1]}"
)
print("Resetting")
instances_on_the_fly.reset()
print(
    f"Instance: {instances_on_the_fly.instance_set[0][0]}, {instances_on_the_fly.instance_set[0][1]}"
)
print("Resetting")
instances_on_the_fly.reset()
print(
    f"Instance: {instances_on_the_fly.instance_set[0][0]}, {instances_on_the_fly.instance_set[0][1]}"
)
print("Resetting")
print("\n")

# Advanced option: directly setting the instance set during training
env = bench.get_environment()
print("Replacing the instance_set mid training")
env.instance_set = {0: [0, 0]}
print_instance_set(env.instance_set)
print("Instance set change")
env.instance_set = {0: [2, 1], 1: [3, 5], 2: [1, 1]}
print_instance_set(env.instance_set)
