import time
import numpy as np
from dacbench.benchmarks import AutoRLBenchmark

single_env_ant =  {
            "lr": 2.5e-4,
            "num_envs": 64,
            "total_timesteps": 1e6,
            "learning_starts": 1000,
            "target_network_update_freq": 500,
            "buffer_size": 1e6,
            "train_frequency": 1,
            "batch_size": 32,
            "epsilon": 0.1,
            "gamma": 0.9,
            "prio_epsilon": 1e-4,
            "alpha": 1.0,
            "beta": 1.0,
            "tau": 0.01,
            "buffer_epsilon": 1e-4,
            "prioritize_replay": False,
            "target": False,
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
            "buffer_size": 1e6,
            "beta": 1.0,
            "env_framework": "gymnax"}

bench = AutoRLBenchmark()
bench.config.instance_set = {0: single_env_ant}
bench.config.algorithm = 'ppo'
env = bench.get_environment()
env.reset()
start = time.time()
_, reward, _, _, _ = env.step({})
end = time.time() - start
print(f"CartPole took {np.round(end, decimals=2)}s for {reward} points in 500000 steps in the gym version.")
