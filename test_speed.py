import time
import numpy as np
from dacbench.benchmarks import AutoRLBenchmark
single_env_cartpole =  {
            "lr": 2.5e-4,
            "num_envs": 4,
            "num_steps": 128,
            "total_timesteps": 1e6,
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

single_env_cartpole_dynamic =  {
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

single_env_ant =  {
            "lr": 2.5e-4,
            "num_envs": 4,
            "num_steps": 128,
            "total_timesteps": 1e6,
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

single_env_ant_dynamic =  {
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

single_env_minatar =  {
            "lr": 2.5e-4,
            "num_envs": 4,
            "total_timesteps": 1e7,
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
            "env_name": "Asterix-MinAtar", 
            "num_eval_episodes": 10,
            "env_framework": "gymnax"}

single_env_minatar_dynamic =  {
            "lr": 2.5e-4,
            "num_envs": 4,
            "total_timesteps": 5e5,
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
            "env_name": "Asterix-MinAtar", 
            "num_eval_episodes": 10,
            "env_framework": "gymnax"}

single_env_minigrid =  {
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
            "env_name": "MiniGrid-DoorKey-5x5-v0", 
            "num_eval_episodes": 10,
            "env_framework": "gym"}

single_env_minigrid_dynamic =  {
            "lr": 2.5e-4,
            "num_envs": 4,
            "total_timesteps": 5e4,
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
            "env_name": "MiniGrid-DoorKey-5x5-v0", 
            "num_eval_episodes": 10,
            "env_framework": "gym"}

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
bench.config.instance_set = {0: single_env_minigrid}
bench.config.checkpoint = ["policy"]
bench.config.track_traj = False
bench.config.checkpoint_dir = "autorl_benchmarks/single_env_level_2"
bench.config.grad_obs = False
bench.config.cutoff = 1
bench.config.algorithm = "dqn"
bench.save_config("dacbench/additional_configs/autorl/single_env_level_2.json")

bench = AutoRLBenchmark()
bench.config.instance_set = {0: single_env_minigrid_dynamic}
bench.config.checkpoint = ["policy"]
bench.config.track_traj = False
bench.config.checkpoint_dir = "autorl_benchmarks/single_env_level_2_dynamic"
bench.config.grad_obs = False
bench.config.cutoff = 20
bench.config.algorithm = "dqn"
bench.save_config("dacbench/additional_configs/autorl/single_env_level_2_dynamic.json")

bench = AutoRLBenchmark()
bench.config.instance_set = {0: single_env_minigrid}
bench.config.checkpoint = ["policy"]
bench.config.track_traj = False
bench.config.checkpoint_dir = "autorl_benchmarks/single_env_level_2_test"
bench.config.grad_obs = False
bench.config.cutoff = 1
bench.config.algorithm = "dqn"
bench.save_config("dacbench/additional_configs/autorl/single_env_level_2_test.json")

bench = AutoRLBenchmark()
bench.config.instance_set = {0: single_env_minigrid_dynamic}
bench.config.checkpoint = ["policy"]
bench.config.track_traj = False
bench.config.checkpoint_dir = "autorl_benchmarks/single_env_level_2_dynamic_test"
bench.config.grad_obs = False
bench.config.cutoff = 20
bench.config.algorithm = "dqn"
bench.save_config("dacbench/additional_configs/autorl/single_env_level_2_dynamic_test.json")


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