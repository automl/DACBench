import argparse
import os

import gym
from collections import OrderedDict
import sys
sys.path.append(os.path.dirname(__file__))
import shutil
from pprint import pprint

from ddqn_local.ddqn import DQN
from utils import make_env, read_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", "-o", type=str, default="output", help='output folder')
    parser.add_argument("--setting-file", "-s", type=str, help='yml file with all settings')
    args = parser.parse_args()

    config_yml_fn = args.setting_file
    exp_params, bench_params, agent_params, train_env_params, eval_env_params = read_config(config_yml_fn)

    if exp_params['n_cores']>1:
        print("WARNING: n_cores>1 is not yet supported")

    # create output folder
    out_dir = args.out_dir
    if os.path.isdir(out_dir) is False:
        os.mkdir(out_dir)
        shutil.copyfile(args.setting_file, out_dir+'/config.yml')

    # get state_dim and action_dim
    temp_env = make_env(bench_params, train_env_params)
    s = temp_env.reset()
    print(temp_env.observation_space) #DEBUG
    print(s) #DEBUG
    state_dim = len(s)
    if isinstance(temp_env.action_space, gym.spaces.Discrete):
        action_dim = temp_env.action_space.n
    else:
        action_dim = temp_env.action_space.shape[0]

    # start the training for ddqn_local agent
    if agent_params['name'] == 'ddqn_local':
        # create train_env and eval_env
        train_env = make_env(bench_params, train_env_params)
        eval_env = make_env(bench_params, eval_env_params)

        assert agent_params['name'] in ['ddqn_local','ppo'], "ERROR: agent " + agent_params['name'] + " is not supported"

        # start ddqn training
        if agent_params['name'] == 'ddqn_local':
            agent = DQN(state_dim, action_dim, agent_params['gamma'], env=train_env, eval_env=eval_env, out_dir=out_dir)
            agent.train(episodes=exp_params['n_episodes'],
                        max_env_time_steps=int(1e9),
                        epsilon=agent_params['epsilon'],
                        eval_eps=exp_params['eval_n_episodes'],
                        eval_every_n_steps=exp_params['eval_interval'],
                        max_train_time_steps=exp_params['n_steps'],
                        begin_learning_after=agent_params['begin_learning_after'],
                        batch_size=agent_params['batch_size'],
                        log_level=exp_params['log_level'],
                        save_best=True,
                        save_model_interval=exp_params['save_interval'])
            agent.save_model('final')
    
    else:
        print(f"Sorry, agent {agent_params['name']} is not yet supported")

main()
