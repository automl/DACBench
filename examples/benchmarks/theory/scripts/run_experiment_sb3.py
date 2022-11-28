import argparse
import os

import gym
import sys

sys.path.append(os.path.dirname(__file__))
import shutil
from utils import make_env, read_config

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
#from stable_baselines3.common.utils import EvalCallback
from eval_callback import LeadingOnesEvalCallback

import torch as th

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", "-o", type=str, default="output", help="output folder"
    )
    parser.add_argument(
        "--setting-file", "-s", type=str, help="yml file with all settings"
    )
    args = parser.parse_args()

    config_yml_fn = args.setting_file
    (
        exp_params,
        bench_params,
        agent_params,
        train_env_params,
        eval_env_params,
    ) = read_config(config_yml_fn)

    if exp_params["n_cores"] > 1:
        print("WARNING: n_cores>1 is not yet supported")

    # create output folder
    out_dir = args.out_dir
    log_dir = f"{out_dir}/logs"
    tb_log_dir = f"{out_dir}/tb_logs"
    if os.path.isdir(out_dir) is False:
        os.mkdir(out_dir)
        shutil.copyfile(args.setting_file, out_dir + "/config.yml")

    if "use_formula" in exp_params:
        if exp_params["use_formula"]:
            print("Using formula for evaluation (instead of running the algorithm itself)")
    else:
        exp["use_formula"] = False

    train_env = make_env(bench_params, train_env_params, test_env=False)
    eval_env = make_env(bench_params, eval_env_params, test_env=True)
    eval_callback = LeadingOnesEvalCallback(eval_env, 
                                use_formula=exp_params["use_formula"],
                                best_model_save_path=out_dir, 
                                log_path=log_dir, 
                                eval_freq=exp_params["eval_interval"],
                                deterministic=True,
                                render=False)
   
    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[50,50],
                     optimizer_class=th.optim.Adam)
    

    agent_name = agent_params["name"]
    assert agent_name in ["DQN","PPO"]

    params = agent_params.copy()
    del params["name"]
    
    agent = globals()[agent_name]("MlpPolicy", train_env, learning_rate=0.001, policy_kwargs=policy_kwargs, verbose=True, tensorboard_log=tb_log_dir, **params)
    print(f"Training with {agent_params['name']}...")
    #print(agent.__dict__)
    agent.learn(total_timesteps=exp_params["n_steps"],
                callback=eval_callback,
                progress_bar=True)


main()
