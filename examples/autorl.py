from dacbench.benchmarks import AutoRLBenchmark
from dacbench.runner import run_benchmark
from dacbench.agents import StaticAgent

import numpy as np

bench = AutoRLBenchmark()

# Get static version of CASH "CASH is King" benchmark
env = bench.get_benchmark(name="tune_me_up", seed=0, dynamic=False, level=1, test=False)
checkpoint_dir = bench.config.checkpoint_dir

# Try a few learning rates on train setting
results = []
lrs = [0.1, 0.01, 0.001]
for lr in lrs:
    env.checkpoint_dir = checkpoint_dir + f"/lr_{lr}"
    action = {"lr": lr}
    agent = StaticAgent(env, action)
    result = run_benchmark(env, agent, 1)
    results.append(result)

# Select best learning rate
best_lr = lrs[np.argmin(results)]
best_action = {"lr": best_lr}
best_agent = StaticAgent(env, best_action)

# Test on test setting
bench = AutoRLBenchmark()

# Evaluate every test seed
rewards = []
for seed in bench.test_seeds:
    env = bench.get_benchmark(name="single_env", seed=seed, dynamic=False, level=1, test=True)
    env.checkpoint_dir = bench.config.checkpoint_dir + f"_seed_{seed}"
    result = run_benchmark(env, best_agent, 1)
    rewards.append(result)

print(f"Mean test reward {-np.mean(rewards)} for learning rate {best_lr}.")