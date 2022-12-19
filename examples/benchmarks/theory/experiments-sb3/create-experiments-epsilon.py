import os
import yaml
import sys
import numpy as np

scriptDir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(f"{scriptDir}/../scripts/")
from utils import default_exp_params, default_bench_params, default_eval_env_params

exp_home_dir = "ddqn-epsilon"

if os.path.isdir(exp_home_dir) is False:
    os.mkdir(exp_home_dir)

with open("run.pbs", "rt") as f:
    pbs_content = "".join(f.readlines())

gamma = 0.95
loss_function = "smooth_l1_loss"

cmds = []
for n in [50, 100, 150, 200][:2]:
    #for k in [3,4,5,6,7,8,10,15,20]:
    for k in [3, 5, 10, 15, 20]:
        for alias in ["evenly_spread","powers_of_2","initial_segment"][:1]:
            for exploration_initial_eps in [0.2, 0.5, 1]:
                for exploration_final_eps in [0.05, 0.1, 0.2]:
                    for exploration_fraction in [0.1, 0.5, 1]:

                        # create experiment folder
                        exp_dir = f"{exp_home_dir}/n{n}-k{k}-{alias}-{exploration_initial_eps}-{exploration_final_eps}-{exploration_fraction}"
                        if os.path.isdir(exp_dir) is False:
                            os.mkdir(exp_dir)

                        # bench config
                        bench_params = default_bench_params
                        if alias == "evenly_spread":
                            action_choices = [i * int(n/k) + 1 for i in range(k)]
                        elif alias == "powers_of_2":
                            action_choices = [2**i for i in range(k)]
                        else:
                            action_choices = list(range(1,k+1))
                        instance_set_path = f"lo_rls_{n}_random.csv"
                        bench_params["alias"] = alias
                        bench_params["action_choices"] = action_choices
                        bench_params["instance_set_path"] = instance_set_path

                        # agent config
                        agent_params = {"name": "DQN",
                                        "gamma": gamma,
                                        "loss_function": loss_function,
                                        "learning_starts": 10_000,
                                        "batch_size": 2048,
                                        "exploration_fraction": exploration_fraction,
                                        "exploration_initial_eps": exploration_initial_eps,
                                        "exploration_final_eps": exploration_final_eps,
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

                        extra = ";".join([
                            "conda init bash",
                            "source activate dacbench",
                            "export PYTHONPATH=$HOME/nguyen/DACBench:$PYTHONPATH",
                            "export PYTHONPATH=$HOME/nguyen/stable-baselines3/:$PYTHONPATH",
                            "cd /home/doerr/nguyen/DACBench/examples/benchmarks/theory/experiments-sb3/"])
                        run_script = os.path.abspath(f"{scriptDir}/../scripts/run_experiment_sb3.py")
                        cmd = f"{extra}; python {run_script} -s {os.path.abspath(exp_dir)}/config.yml -o {os.path.abspath(exp_dir)} 2>&1 | tee {os.path.abspath(exp_dir)}/out"
                        cmds.append(cmd)

print("\n".join(cmds))
