import argparse
import os

import gym
import sys

sys.path.append(os.path.dirname(__file__))
import shutil
import time

from ddqn_local.ddqn import DQN
from ddqn_local.tdqn import TDQN as TDQN
from ddqn_local.tdqn_original import TDQN as TDQNOriginal
from utils import make_env, read_config

from torch.nn import functional as F

def main():
    start_time = time.time()
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
    log_dir = f"{out_dir}"
    tb_log_dir = f"{out_dir}/tb_logs"
    if os.path.isdir(out_dir) is False:
        os.mkdir(out_dir)
        shutil.copyfile(args.setting_file, out_dir + "/config.yml")

    if exp_params["use_formula"]:
        print("Using formula for evaluation (instead of running the algorithm itself)")
    
    # get state_dim and action_dim
    temp_env = make_env(bench_params, train_env_params)
    s = temp_env.reset()
    print(temp_env.observation_space)  # DEBUG
    print(s)  # DEBUG
    state_dim = len(s)
    if isinstance(temp_env.action_space, gym.spaces.Discrete):
        action_dim = temp_env.action_space.n
    else:
        action_dim = temp_env.action_space.shape[0]

    # create train_env and eval_env
    train_env = make_env(bench_params, train_env_params)
    eval_env = make_env(bench_params, eval_env_params)

    # get loss function
    assert agent_params["loss_function"] in ["mse_loss", "smooth_l1_loss"]
    loss_function = getattr(F, agent_params["loss_function"])

    # start the training for ddqn_local agent
    if agent_params["name"] == "ddqn_local":
        agent = DQN(
            state_dim=state_dim,
            action_dim=action_dim,
            env=train_env,
            eval_env=eval_env,
            out_dir=out_dir,
            gamma=agent_params["gamma"],
            loss_function=loss_function,
        )
        print(agent.__dict__)
        agent.train(
            episodes=exp_params["n_episodes"],
            max_env_time_steps=int(1e9),
            epsilon=agent_params["epsilon"],
            eval_every_n_steps=exp_params["eval_interval"],
            save_agent_at_every_eval=exp_params["save_agent_at_every_eval"],
            n_eval_episodes_per_instance=exp_params["n_eval_episodes_per_instance"],
            max_train_time_steps=exp_params["n_steps"],
            begin_learning_after=agent_params["begin_learning_after"],
            batch_size=agent_params["batch_size"],
            log_level=exp_params["log_level"],
            use_formula=exp_params["use_formula"]
        )

    elif agent_params["name"] == "tdqn_local":
        assert "skip_dim" in agent_params
        assert "skip_batch_size" in agent_params
        agent = TDQN(
            state_dim=state_dim,
            action_dim=action_dim,
            skip_dim=agent_params["skip_dim"],
            env=train_env,
            eval_env=eval_env,
            out_dir=out_dir,
            gamma=agent_params["gamma"],
            loss_function=loss_function,
        )
        print(agent.__dict__)
        agent.train(
            episodes=exp_params["n_episodes"],
            max_env_time_steps=int(1e9),
            epsilon=agent_params["epsilon"],
            eval_every_n_steps=exp_params["eval_interval"],
            save_agent_at_every_eval=exp_params["save_agent_at_every_eval"],
            n_eval_episodes_per_instance=exp_params["n_eval_episodes_per_instance"],
            max_train_time_steps=exp_params["n_steps"],
            begin_learning_after=agent_params["begin_learning_after"],
            batch_size=agent_params["batch_size"],
            skip_batch_size=agent_params["skip_batch_size"],
            log_level=exp_params["log_level"],
            use_formula=exp_params["use_formula"]
        )

    elif agent_params["name"] == "tdqn_original":
        assert "skip_dim" in agent_params
        assert "skip_batch_size" in agent_params
        agent = TDQNOriginal(
            state_dim=state_dim,
            action_dim=action_dim,
            skip_dim=agent_params["skip_dim"],
            env=train_env,
            eval_env=eval_env,
            out_dir=out_dir,
            gamma=agent_params["gamma"],
            loss_function=loss_function,
        )
        print(agent.__dict__)
        agent.train(
            episodes=exp_params["n_episodes"],
            max_env_time_steps=int(1e9),
            epsilon=agent_params["epsilon"],
            eval_every_n_steps=exp_params["eval_interval"],
            save_agent_at_every_eval=exp_params["save_agent_at_every_eval"],
            n_eval_episodes_per_instance=exp_params["n_eval_episodes_per_instance"],
            max_train_time_steps=exp_params["n_steps"],
            begin_learning_after=agent_params["begin_learning_after"],
            batch_size=agent_params["batch_size"],
            skip_batch_size=agent_params["skip_batch_size"],
            log_level=exp_params["log_level"],
            use_formula=exp_params["use_formula"]
        )

    else:
        print(f"Sorry, agent {agent_params['name']} is not yet supported")
    
    total_time = time.time() - start_time
    print(f"Total runtime: {total_time}")

main()
