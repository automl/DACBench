import os
import yaml
import sys
import numpy as np

scriptDir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(f"{scriptDir}/../scripts/")
from utils import default_exp_params, default_bench_params, default_eval_env_params

exp_home_dir = "cross-instances"
gamma = 0.95
loss_function = "smooth_l1_loss"

if os.path.isdir(exp_home_dir) is False:
    os.mkdir(exp_home_dir)

with open("run.pbs", "rt") as f:
    pbs_content = "".join(f.readlines())


#for k in [3,4,5,6,7,8,10,15,20]:
for k in [3,5,10,15,20]:
    for alias in ["evenly_spread","powers_of_2","initial_segment"][:1]:
        for instance_set_path in ["lo_rls_50_random.csv", "lo_rls_100_random.csv", "lo_rls_40_60_80_random.csv", "lo_rls_40_60_random.csv"]:

            # create experiment folder
            s_train_sizes = instance_set_path.replace("lo_rls_","").replace("_random.csv","")
            train_sizes = [int(s) for s in s_train_sizes.split("_")]
            exp_dir = f"{exp_home_dir}/k{k}-{alias}-{s_train_sizes}"
            if os.path.isdir(exp_dir) is False:
                os.mkdir(exp_dir)

            # bench config
            bench_params = default_bench_params
            n = max(train_sizes)
            if alias == "evenly_spread":
                action_choices = [i * int(n/k) + 1 for i in range(k)]
            elif alias == "powers_of_2":
                action_choices = [2**i for i in range(k)]
            else:
                action_choices = list(range(1,k+1))
            bench_params["alias"] = alias
            bench_params["action_choices"] = action_choices
            bench_params["instance_set_path"] = instance_set_path

            # agent config
            agent_params = {"name": "DQN",
                            "gamma": gamma,
                            "loss_function": loss_function,
                            "learning_starts": 10_000,
                            "batch_size": 2048,
                            "exploration_fraction": 1,
                            "exploration_initial_eps": 0.2,
                            "exploration_final_eps": 0.2,
                            "max_grad_norm": 1e9,
                            "tau": 0.01,
                            "learning_rate": 0.001,
                            "train_freq": 1,
                            "gradient_steps": 1,
                            "target_update_interval": 1
            }
            
            # write config to yaml file
            config = {"experiment": default_exp_params,
                      "bench": bench_params,
                      "eval_env": default_eval_env_params,
                      "agent": agent_params}
            config_file = f"{exp_dir}/config.yml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f, explicit_start=True)

            # run.pbs
            pbs_file = f"{exp_dir}/run.pbs"
            with open(pbs_file, "wt") as f:
                f.write(pbs_content.replace("<exp_dir>", exp_dir))

            # qsub command
            cmd = f"qsub -N {os.path.basename(exp_dir)} {pbs_file}"
            print(cmd)
            
