import time
import numpy as np
from dacbench.benchmarks import AutoRLBenchmark

config_cartpole =  {
            "lr": 2.5e-4,
            "num_envs": 4,
            "num_steps": 128,
            "total_timesteps": 5e4,
            "update_epochs": 4,
            "num_minibatches": 4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "activation": "tanh",
            "hidden_size": 64,
            "env_name": "CartPole-v1", 
            "num_eval_episodes": 10}

config_cartpole_full =  {
            "lr": 2.5e-4,
            "num_envs": 4,
            "num_steps": 128,
            "total_timesteps": 5e5,
            "update_epochs": 4,
            "num_minibatches": 4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "activation": "tanh",
            "hidden_size": 64,
            "env_name": "CartPole-v1", 
            "num_eval_episodes": 10}

config_minatar_breakout =  {
            "lr": 5e-4,
            "num_envs": 64,
            "num_steps": 128,
            "total_timesteps": 1e6,
            "update_epochs": 4,
            "num_minibatches": 8,
            "gamma": 0.999,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "activation": "tanh",
            "hidden_size": 256,
            "env_name": "Breakout-MinAtar", 
            "num_eval_episodes": 10}

config_minatar_full =  {
            "lr": 5e-4,
            "num_envs": 64,
            "num_steps": 128,
            "total_timesteps": 1e7,
            "update_epochs": 4,
            "num_minibatches": 8,
            "gamma": 0.999,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "activation": "tanh",
            "hidden_size": 256,
            "env_name": "Breakout-MinAtar", 
            "num_eval_episodes": 10}

bench = AutoRLBenchmark()
bench.config.instance_set = {0: config_cartpole_full}
env = bench.get_environment()
env.reset()
start = time.time()
_, reward, _, _, _ = env.step({})
end = time.time() - start
print(f"CartPole took {np.round(end, decimals=2)}s for {reward} points in 100000 steps.")

bench = AutoRLBenchmark()
bench.config.instance_set = {0: config_cartpole}
env = bench.get_environment()
env.reset()
start = time.time()
for i in range(10):#563):
    _, reward, _, _, _ = env.step({})
    if i == 0:
        end = time.time() - start
        print(f"CartPole took {np.round(end, decimals=2)}s for {reward} points in 64 steps.")
end = time.time() - start
print(f"CartPole took {np.round(end, decimals=2)}s for {reward} points in 100000 steps in 10 intervals of 10000 steps.")

bench = AutoRLBenchmark()
bench.config.instance_set = {0: config_minatar_full}
env = bench.get_environment()
env.reset()
start = time.time()
_, reward, _, _, _ = env.step({})
end = time.time() - start
print(f"MinAtar Breakout took {np.round(end, decimals=2)}s for {reward} points in 10M steps.")

bench = AutoRLBenchmark()
bench.config.instance_set = {0: config_minatar_breakout}
env = bench.get_environment()
env.reset()
start = time.time()
for i in range(10):#000):
    _, reward, _, _, _ = env.step({})
    if i == 0:
        end = time.time() - start
        print(f"MinAtar Breakout took {np.round(end, decimals=2)}s for {reward} points in 1000 steps.")
end = time.time() - start
print(f"MinAtar Breakout took {np.round(end, decimals=2)}s for {reward} points in 10M steps in 10 intervals of 1e6 steps.")