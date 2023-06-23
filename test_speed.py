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
            "num_eval_episodes": 10,
            "env_framework": "gymnax"}

config_cartpole_dqn =  {
            "lr": 2.5e-4,
            "num_envs": 4,
            "total_timesteps": 1e6,
            "learning_starts": 1000,
            "target_network_update_freq": 500,
            "buffer_size": 1e6,
            "train_frequency": 1,
            "batch_size": 32,
            "epsilon": 0.1,
            "gamma": 0.9,
            "target": False,
            "activation": "tanh",
            "hidden_size": 64,
            "env_name": "CartPole-v1", 
            "num_eval_episodes": 10,
            "env_framework": "gymnax",
            "checkpoint_dir": "test_cartpole"}

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
            "env_name": "BreakoutDeterministic-v4", 
            "num_eval_episodes": 10,
            "env_framework": "gym"}

config_minatar_full =  {
            "lr": 5e-4,
            "num_envs": 64,
            "num_steps": 128,
            "total_timesteps": 25e6,
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
            "env_name": "procgen:procgen-coinrun-v0", 
            "num_eval_episodes": 10,
            "env_framework": "gym",
            "track_traj": False}

bench = AutoRLBenchmark()
bench.config.instance_set = {0: config_cartpole_dqn}
bench.config.checkpoint = True
bench.config.checkpoint_dir = "test_cartpole"
bench.config.grad_obs = True
bench.config.track_trajectory = True
bench.config.cutoff = 1
bench.config.algorithm = "dqn"
env = bench.get_environment()
env.reset()
start = time.time()
_, reward, te, tr, _ = env.step({})
end = time.time() - start
print(te or tr)
print(f"CartPole took {np.round(end, decimals=2)}s for {reward} points in 1M steps in the gymnax version (including saving all data).")

# bench = AutoRLBenchmark()
# bench.config.instance_set = {0: config_cartpole_full}
# bench.config.env_framework = "gym"
# env = bench.get_environment()
# env.reset()
# start = time.time()
# _, reward, _, _, _ = env.step({})
# end = time.time() - start
# print(f"CartPole took {np.round(end, decimals=2)}s for {reward} points in 500000 steps in the gym version.")


# bench = AutoRLBenchmark()
# bench.config.instance_set = {0: config_cartpole}
# env = bench.get_environment()
# env.reset()
# start = time.time()
# for i in range(10):#563):
#     _, reward, _, _, _ = env.step({})
#     if i == 0:
#         end = time.time() - start
#         print(f"CartPole took {np.round(end, decimals=2)}s for {reward} points in 64 steps.")
# end = time.time() - start
# print(f"CartPole took {np.round(end, decimals=2)}s for {reward} points in 100000 steps in 10 intervals of 10000 steps.")


# bench = AutoRLBenchmark()
# bench.config.instance_set = {0: config_minatar_full}
# env = bench.get_environment()
# env.reset()
# start = time.time()
# for i in range(int(1)):
#     _, reward, _, _, _ = env.step({})
#     if i == 0:
#         end = time.time() - start
#         print(f"ProcGen took {np.round(end, decimals=2)}s for {reward} points in 1.000.000 steps.")
# end = time.time() - start
# print(f"ProcGen took {np.round(end, decimals=2)}s for {reward} points in 25M steps in 25 intervals of 1.000.000 steps.")

# bench = AutoRLBenchmark()
# bench.config.instance_set = {0: config_minatar_full}
# env = bench.get_environment()
# env.reset()
# start = time.time()
# _, reward, _, _, _ = env.step({})
# end = time.time() - start
# print(f"Breakout took {np.round(end, decimals=2)}s for {reward} points in 10M steps.")