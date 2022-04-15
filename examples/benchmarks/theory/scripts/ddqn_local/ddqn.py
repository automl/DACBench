import pickle
import gzip

import scipy.signal
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import count
from collections import namedtuple
import time
import sys
import os
import json

sys.path.append(os.path.dirname(__file__))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tt(ndarray):
    """
    Helper Function to cast observation to correct type/device
    """
    if device == "cuda":
        return Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)
    else:
        return Variable(torch.from_numpy(ndarray).float(), requires_grad=False)


def soft_update(target, source, tau):
    """
    Simple Helper for updating target-network parameters
    :param target: target network
    :param source: policy network
    :param tau: weight to regulate how strongly to update (1 -> copy over weights)
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    """
    See soft_update
    """
    soft_update(target, source, 1.0)


class Q(nn.Module):
    """
    Simple fully connected Q function. Also used for skip-Q when concatenating behaviour action and state together.
    Used for simpler environments such as mountain-car or lunar-lander.
    """

    def __init__(self, state_dim, action_dim, non_linearity=F.relu, hidden_dim=50):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self._non_linearity = non_linearity

    def forward(self, x):
        x = self._non_linearity(self.fc1(x))
        x = self._non_linearity(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """
    Simple Replay Buffer. Used for standard DQN learning.
    """

    def __init__(self, max_size):
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "terminal_flags"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], terminal_flags=[])
        self._size = 0
        self._max_size = max_size

    def add_transition(self, state, action, next_state, reward, done):
        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.terminal_flags.append(done)
        self._size += 1

        if self._size > self._max_size:
            self._data.states.pop(0)
            self._data.actions.pop(0)
            self._data.next_states.pop(0)
            self._data.rewards.pop(0)
            self._data.terminal_flags.pop(0)

    def random_next_batch(self, batch_size):
        batch_indices = np.random.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_terminal_flags = np.array([self._data.terminal_flags[i] for i in batch_indices])
        return tt(batch_states), tt(batch_actions), tt(batch_next_states), tt(batch_rewards), tt(batch_terminal_flags)

    def save(self, path):
        with open(os.path.join(path, 'rpb.pkl'), 'wb') as fh:
            pickle.dump(list(self._data), fh)

    def load(self, path):
        with open(os.path.join(path, 'rpb.pkl'), 'rb') as fh:
            data = pickle.load(fh)
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "terminal_flags"])
        self._data.states = data[0]
        self._data.actions = data[1]
        self._data.next_states = data[2]
        self._data.rewards = data[3]
        self._data.terminal_flags = data[4]
        self._size = len(data[0])


class DQN:
    """
    Simple double DQN Agent
    """

    def __init__(self, state_dim: int, action_dim: int, gamma: float,
                 env: gym.Env, eval_env: gym.Env, train_eval_env: gym.Env = None, vision: bool = False, out_dir:str = "./output/"):
                 
        """
        Initialize the DQN Agent
        :param state_dim: dimensionality of the input states
        :param action_dim: dimensionality of the output actions
        :param gamma: discount factor
        :param env: environment to train on
        :param eval_env: environment to evaluate on
        :param eval_env: environment to evaluate on with training data
        :param vision: boolean flag to indicate if the input state is an image or not
        """
        if not vision:  # For featurized states
            self._q = Q(state_dim, action_dim).to(device)
            self._q_target = Q(state_dim, action_dim).to(device)
        else:  # For image states, i.e. Atari
            raise NotImplementedError

        self._eval_r_best = None

        self._gamma = gamma
        self._loss_function = nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.001)
        self._action_dim = action_dim

        self._replay_buffer = ReplayBuffer(1e6)
        self._env = env
        self._eval_env = eval_env
        self._train_eval_env = train_eval_env
        self.out_dir = out_dir


    def save_rpb(self, path):
        self._replay_buffer.save(path)

    def load_rpb(self, path):
        self._replay_buffer.load(path)

    def get_action(self, x: np.ndarray, epsilon: float) -> int:
        """
        Simple helper to get action epsilon-greedy based on observation x
        """
        u = np.argmax(self._q(tt(x)).detach().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._action_dim)
        return u


    def train(self, episodes: int, max_env_time_steps: int, epsilon: float, eval_eps: int = 1,
              eval_every_n_steps: int = 1, max_train_time_steps: int = 1_000_000, 
              begin_learning_after: int = 10_000, batch_size: int = 2_048,
              log_level=1, save_best=True, save_model_interval=1000):
        """
        Training loop
        :param episodes: maximum number of episodes to train for
        :param max_env_time_steps: maximum number of steps in the environment to perform per episode
        :param epsilon: constant epsilon for exploration when selecting actions
        :param eval_eps: numper of episodes to run for evaluation
        :param eval_every_n_steps: interval of steps after which to evaluate the trained agent
        :param max_train_time_steps: maximum number of steps to train
        :param log_level: 
            (1) basic log: evaluation scores and evaluation episodes' infos
            (2) extensive log: (1) and discounted rewards per episode (evaluation), average value-action estimate per episode (evaluation), average discounted rewards per episode (evaluation), average loss per batch 
        :return:
        """
        total_steps = 0
        start_time = time.time()
        def write_info(obj, mode='a+'):
            with gzip.open(os.path.join(self.out_dir,'eval_infos.gzip'),mode) as f:
                pickle.dump(obj,f)            
        def write_extra_log(obj, mode='a+'):
            with gzip.open(os.path.join(self.out_dir,'eval_extra_log.gzip'),mode) as f:
                pickle.dump(obj,f)
        def write_train_log(obj, mode='a+'):
            with gzip.open(os.path.join(self.out_dir,'train_log.gzip'),mode) as f:
                pickle.dump(obj,f)
        for e in range(episodes):
            # print('\033c')
            # print('\x1bc')
            #if e % 100 == 0:
            #    print("%s/%s" % (e + 1, episodes)) 
            s = self._env.reset()
            ep_loss = []
            for t in range(max_env_time_steps):
                a = self.get_action(s, epsilon if total_steps > begin_learning_after else 1.)
                ns, r, d, _ = self._env.step(a)                
                total_steps += 1
                print("#episodes=%d, #steps=%d" % (e+1, total_steps), end='\r')

                ########### Begin Evaluation
                if (total_steps % eval_every_n_steps) == 0:
                    #print('Begin Evaluation')
                    eval_s, eval_r, eval_d, infos, eval_extra_log = self.eval(episodes=eval_eps, max_env_time_steps=max_env_time_steps, train_set=False, log_level=log_level) 
                    eval_stats = dict(
                        elapsed_time=time.time() - start_time,
                        training_steps=total_steps,
                        training_eps=e,
                        avg_num_steps_per_eval_ep=float(np.mean(eval_s)),
                        avg_num_decs_per_eval_ep=float(np.mean(eval_d)),
                        avg_rew_per_eval_ep=float(np.mean(eval_r)),
                        std_rew_per_eval_ep=float(np.std(eval_r)),
                        eval_eps=eval_eps
                    )
                    per_inst_stats = dict(
                            # eval_insts=self._train_eval_env.instances,
                            reward_per_isnts=eval_r,
                            steps_per_insts=eval_s,
                        )
                    with open(os.path.join(self.out_dir, 'eval_scores.json'), 'a+') as out_fh:
                        json.dump(eval_stats, out_fh)
                        out_fh.write('\n')
                    with open(os.path.join(self.out_dir, 'eval_scores_per_inst.json'), 'a+') as out_fh:
                        json.dump(per_inst_stats, out_fh)
                        out_fh.write('\n')
                    #with open(os.path.join(self.out_dir, 'eval_infos_per_episode.json'), 'a+') as out_fh:
                    #    json.dump(infos, out_fh)
                    #    out_fh.write('\n')
                    write_info(infos)
                    #pickle.dump(infos, eval_infos_file)
                    if log_level>1:
                        #pickle.dump(eval_extra_log, eval_extra_log_file)
                        write_extra_log(eval_extra_log)

                    # save best model
                    if (self._eval_r_best is None) or (self._eval_r_best < eval_stats['avg_rew_per_eval_ep']):
                        self._eval_r_best = eval_stats['avg_rew_per_eval_ep']
                        self.save_model('best')

                    if self._train_eval_env is not None:
                        eval_s, eval_r, eval_d, pols, infos = self.eval(eval_eps, max_env_time_steps, train_set=True)
                        eval_stats = dict(
                            elapsed_time=time.time() - start_time,
                            training_steps=total_steps,
                            training_eps=e,
                            avg_num_steps_per_eval_ep=float(np.mean(eval_s)),
                            avg_num_decs_per_eval_ep=float(np.mean(eval_d)),
                            avg_rew_per_eval_ep=float(np.mean(eval_r)),
                            std_rew_per_eval_ep=float(np.std(eval_r)),
                            eval_eps=eval_eps
                        )
                        per_inst_stats = dict(
                            # eval_insts=self._train_eval_env.instances,
                            reward_per_isnts=eval_r,
                            steps_per_insts=eval_s,

                        )

                        with open(os.path.join(self.out_dir, 'train_scores.json'), 'a+') as out_fh:
                            json.dump(eval_stats, out_fh)
                            out_fh.write('\n')
                        with open(os.path.join(self.out_dir, 'train_scores_per_inst.json'), 'a+') as out_fh:
                            json.dump(per_inst_stats, out_fh)
                            out_fh.write('\n')
                        with open(os.path.join(self.out_dir, 'train_infos_per_episode.json'), 'a+') as out_fh:
                            json.dump(infos, out_fh)
                            out_fh.write('\n')
                    #print('End Evaluation')
                    print('\n(eval) R=%.2f' % (float(np.mean(eval_r))))
                ########### End Evaluation

                # save checkpoint models
                if (total_steps % save_model_interval) == 0:
                    self.save_model("checkpoint_" + str(total_steps))

                # Update replay buffer
                self._replay_buffer.add_transition(s, a, ns, r, d)
                if begin_learning_after < total_steps:
                    batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = \
                        self._replay_buffer.random_next_batch(batch_size)

                    ########### Begin double Q-learning update
                    target = batch_rewards + (1 - batch_terminal_flags) * self._gamma * \
                             self._q_target(batch_next_states)[torch.arange(batch_size).long(), torch.argmax(
                                 self._q(batch_next_states), dim=1)]
                    current_prediction = self._q(batch_states)[torch.arange(batch_size).long(), batch_actions.long()]

                    loss = self._loss_function(current_prediction, target.detach())
                    if log_level>1:
                        ep_loss.append(loss.detach().numpy())

                    self._q_optimizer.zero_grad()
                    loss.backward()
                    self._q_optimizer.step()

                    soft_update(self._q_target, self._q, 0.01)
                    ########### End double Q-learning update

                if d:
                    break
                s = ns
                if total_steps >= max_train_time_steps:
                    break
            
            if log_level>1:
                if len(ep_loss)>0:
                    #pickle.dump({'ep_loss':ep_loss, 'total_steps': total_steps}, train_log_file)                
                    write_train_log({'ep_loss':ep_loss, 'total_steps': total_steps})
                ep_loss = []
        
            if total_steps >= max_train_time_steps:
                break

        # Final evaluation
        eval_s, eval_r, eval_d, infos, eval_extra_log = self.eval(episodes=eval_eps, max_env_time_steps=max_env_time_steps, train_set=False, log_level=log_level) 
        eval_stats = dict(
            elapsed_time=time.time() - start_time,
            training_steps=total_steps,
            training_eps=e,
            avg_num_steps_per_eval_ep=float(np.mean(eval_s)),
            avg_num_decs_per_eval_ep=float(np.mean(eval_d)),
            avg_rew_per_eval_ep=float(np.mean(eval_r)),
            std_rew_per_eval_ep=float(np.std(eval_r)),
            eval_eps=eval_eps
        )
        per_inst_stats = dict(
                # eval_insts=self._train_eval_env.instances,
                reward_per_isnts=eval_r,
                steps_per_insts=eval_s
            )

        with open(os.path.join(self.out_dir, 'eval_scores.json'), 'a+') as out_fh:
            json.dump(eval_stats, out_fh)
            out_fh.write('\n')
        with open(os.path.join(self.out_dir, 'eval_scores_per_inst.json'), 'a+') as out_fh:
            json.dump(per_inst_stats, out_fh)
            out_fh.write('\n')
        #with open(os.path.join(self.out_dir, 'eval_infos_per_episode.json'), 'a+') as out_fh:
        #    json.dump(infos, out_fh)
        #    out_fh.write('\n')
        #pickle.dump(infos, eval_infos_file)
        write_info(infos)
        if log_level>1:
            #pickle.dump(eval_extra_log, eval_extra_log_file)
            write_extra_log(eval_extra_log)
        #eval_infos_file.close()
        #eval_extra_log_file.close()
        #train_log_file.close()

        if self._train_eval_env is not None:
            eval_s, eval_r, eval_d, pols, infos = self.eval(eval_eps, max_env_time_steps, train_set=True)
            eval_stats = dict(
                elapsed_time=time.time() - start_time,
                training_steps=total_steps,
                training_eps=e,
                avg_num_steps_per_eval_ep=float(np.mean(eval_s)),
                avg_num_decs_per_eval_ep=float(np.mean(eval_d)),
                avg_rew_per_eval_ep=float(np.mean(eval_r)),
                std_rew_per_eval_ep=float(np.std(eval_r)),
                eval_eps=eval_eps
            )
            per_inst_stats = dict(
                # eval_insts=self._train_eval_env.instances,
                reward_per_isnts=eval_r,
                steps_per_insts=eval_s,
            )

            with open(os.path.join(self.out_dir, 'train_scores.json'), 'a+') as out_fh:
                json.dump(eval_stats, out_fh)
                out_fh.write('\n')
            with open(os.path.join(self.out_dir, 'train_scores_per_inst.json'), 'a+') as out_fh:
                json.dump(per_inst_stats, out_fh)
                out_fh.write('\n')
            with open(os.path.join(self.out_dir, 'train_infos_per_episode.json'), 'a+') as out_fh:
                json.dump(infos, out_fh)
                out_fh.write('\n')

    def __repr__(self):
        return 'DDQN'

    def eval(self, episodes: int, max_env_time_steps: int, train_set: bool = False, log_level: int = 1):
        """
        Simple method that evaluates the agent with fixed epsilon = 0
        :param episodes: max number of episodes to play
        :param max_env_time_steps: max number of max_env_time_steps to play

        :returns (steps per episode), (reward per episode), (decisions per episode)
        """
        steps, rewards, decisions = [], [], []
        infos = []
        this_env = self._eval_env if not train_set else self._train_eval_env
        extra_log = {'dis_r': [], 'est_q':[], 'avg_dis_r_per_state': [], 'avg_est_qs_per_state': []} # extra log info when log_level>1
        for e in range(episodes):
            # this_env.instance_index = this_env.instance_index % 10  # for faster debugging on only 10 insts
            #print(f'Eval Episode {e} of {episodes}', end='\r')
            ed, es, er = 0, 0, 0

            ep_s, ep_r = [], [] # list of states visited during current episode and the rewards 

            s = this_env.reset()
            info_this_episode = []
            for _ in count():
                with torch.no_grad():
                    a = self.get_action(s, 0)
                    ns, r, d, info = this_env.step(a)
                if log_level > 1:
                    ep_s.append(ns)
                    ep_r.append(r)
                if d: 
                    info_this_episode.append(info)
                ed += 1
                er += r
                es += 1
                if es >= max_env_time_steps or d:
                    break
                s = ns          
            steps.append(es)
            rewards.append(float(er))
            decisions.append(ed)
            infos.append(info_this_episode)

            if log_level > 1:
                # discounted cumulative reward for each visited state
                dis_r_per_state = scipy.signal.lfilter([1], [1, float(-self._gamma)], ep_r[::-1], axis=0)[::-1] # copy from https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/ppo/core.py#L29
                # estimated value for each visited state
                with torch.no_grad():
                    est_qs_per_state = np.amax(self._q(tt(np.array(ep_s))).detach().numpy(), axis=1)
                extra_log['avg_dis_r_per_state'].append(np.mean(dis_r_per_state))
                extra_log['avg_est_qs_per_state'].append(np.mean(est_qs_per_state))
                extra_log['dis_r'].append(dis_r_per_state[0])
                extra_log['est_q'].append(est_qs_per_state[0])

        return steps, rewards, decisions, infos, extra_log

    def save_model(self, model_name):
        torch.save(self._q.state_dict(), os.path.join(self.out_dir, model_name + '.pt'))

    def load(self, path):
        self._q.load_state_dict(torch.load(path))


def train_ddqn_local(train_env, eval_env, state_dim, action_dim, train_episodes=1000, train_max_steps=100000, 
                    eval_episodes=10, eval_interval=1000,
                    begin_learning_after=10000, batch_size=2048, epsilon=0.2, gamma=0.99,
                    out_dir = "./output", save_model_interval=1000, save_best=True, save_final=True, log_level=1):
     
    agent = DQN(state_dim, action_dim, gamma, env=train_env, eval_env=eval_env, out_dir=out_dir)

    agent.train(episodes=train_episodes, 
                max_env_time_steps=int(1e9),
                epsilon=epsilon,
                eval_eps=eval_episodes,
                eval_every_n_steps=eval_interval,
                max_train_time_steps=train_max_steps,
                begin_learning_after=begin_learning_after,
                batch_size=batch_size,
                log_level=log_level,
                save_best=save_best,
                save_model_interval=save_model_interval)

    agent.save_model('final')
