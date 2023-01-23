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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from leadingones_eval import LeadingOnesEval
from ddqn import tt, Q, soft_update, hard_update, ReplayBuffer, DQN

class SkipReplayBuffer:
    """
    Replay Buffer for training the skip-Q.
    Expects "concatenated states" which already contain the behaviour-action for the skip-Q.
    Stores transitions as usual but with additional skip-length. The skip-length is used to properly discount.
    """

    def __init__(self, max_size):
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states",
                                                 "rewards", "terminal_flags", "lengths"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], terminal_flags=[], lengths=[])
        self._size = 0
        self._max_size = max_size

    def add_transition(self, state, action, next_state, reward, done, length):
        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.terminal_flags.append(done)
        self._data.lengths.append(length)  # Observed skip-length of the transition
        self._size += 1

        if self._size > self._max_size:
            self._data.states.pop(0)
            self._data.actions.pop(0)
            self._data.next_states.pop(0)
            self._data.rewards.pop(0)
            self._data.terminal_flags.pop(0)
            self._data.lengths.pop(0)

    def random_next_batch(self, batch_size):
        batch_indices = np.random.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_terminal_flags = np.array([self._data.terminal_flags[i] for i in batch_indices])
        batch_lengths = np.array([self._data.lengths[i] for i in batch_indices])
        return tt(batch_states), tt(batch_actions), tt(batch_next_states),\
               tt(batch_rewards), tt(batch_terminal_flags), tt(batch_lengths)


class TDQN(DQN):
    """
    Simple double DQN Agent
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        skip_dim: int,
        env: gym.Env,
        eval_env: gym.Env,
        train_eval_env: gym.Env = None,
        vision: bool = False,
        out_dir: str = "./output/",
        gamma: float = 0.99,
        loss_function = F.mse_loss
    ):
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
        super(TDQN, self).__init__(state_dim=state_dim, action_dim=action_dim,
                                    env=env, eval_env=eval_env, train_eval_env=train_eval_env,
                                    out_dir=out_dir, gamma=gamma, loss_function=loss_function)

        self._skip_q = Q(state_dim+1, skip_dim).to(device)
        self._skip_loss_function = loss_function
        self._skip_q_optimizer = optim.Adam(self._skip_q.parameters(), lr=0.001)
        self._skip_dim = skip_dim
        self._skip_replay_buffer = SkipReplayBuffer(1e6)

    def get_skip(self, x: np.ndarray, epsilon: float) -> int:
        """
        Simple helper to get skip epsilon-greedy based on observation+action
        """
        u = np.argmax(self._skip_q(tt(x)).detach().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._skip_dim)
        return u

    def train(
        self,
        episodes: int,
        max_env_time_steps: int = 1_000_000,
        epsilon: float = 0.2,
        eval_every_n_steps: int = 1000,
        n_eval_episodes_per_instance: int = 20,
        save_agent_at_every_eval: bool = True,
        max_train_time_steps: int = 1_000_000,
        begin_learning_after: int = 10_000,
        batch_size: int = 2_048,
        skip_batch_size: int = 64,
        log_level=1,
        use_formula=True
    ):
        """
        Training loop
        :param episodes: maximum number of episodes to train for
        :param max_env_time_steps: maximum number of steps in the environment to perform per episode
        :param epsilon: constant epsilon for exploration when selecting actions
        :param eval_every_n_steps: interval of steps after which to evaluate the trained agent
        :param max_train_time_steps: maximum number of steps to train
        :param log_level:
            (1) basic log: evaluation scores and evaluation episodes' infos
            (2) extensive log: (1) and discounted rewards per episode (evaluation), average value-action estimate per episode (evaluation), average discounted rewards per episode (evaluation), average loss per batch
        :return:
        """
        total_steps = 0
        start_time = time.time()

        assert use_formula==False, "Error: TempoRL doesn't support formula yet"

        self.evaluator = LeadingOnesEval(eval_env=self._eval_env,
                                         agent=self,
                                         use_formula=use_formula,
                                         n_eval_episodes_per_instance=n_eval_episodes_per_instance,
                                         log_path=self.out_dir,
                                         save_agent_at_every_eval=save_agent_at_every_eval,
                                         verbose=log_level)

        for e in range(episodes):
            s = self._env.reset()
            ep_loss = []

            done = False
            ep_n_steps = 0
            for t in range(max_env_time_steps):
                if total_steps > begin_learning_after:
                    cur_epsilon = epsilon
                else:
                    cur_epsilon = 1.0

                a = self.get_action(s, cur_epsilon)
                skip_state = np.hstack([s, [a]])  # concatenate action to the state
                skip = self.get_skip(skip_state, cur_epsilon) + 1

                skip_rewards = []
                skip_states = []
                for n_skips in range(1,skip+1): # play the same action "skip" times

                    # make one step
                    ns, r, done, _ = self._env.step(a)
                    total_steps += 1
                    ep_n_steps += 1
                    skip_rewards.append(r)
                    skip_states.append(np.hstack([s, [a]]))

                    # evaluation
                    if (total_steps % eval_every_n_steps) == 0:
                        self.evaluator.eval(total_steps, with_skip=True)

                    # calculate discounted rewards for all skip connections, and update skip replay buffer
                    dis_r = 0 # cummulative discounted reward, starting from the last state
                    for i in range(n_skips): 
                        skip_length = i + 1
                        start_state = skip_states[-skip_length]
                        dis_r = skip_rewards[-skip_length] + self._gamma * dis_r
                        self._skip_replay_buffer.add_transition(start_state, a, ns, dis_r, done, skip_length)

                    # update skip-Q network with Q-network and Q-target network
                    # NGUYEN: we add n_skips transitions to skip buffer but only update once
                    batch_states, batch_actions, batch_next_states, batch_rewards, \
                        batch_terminal_flags, batch_lengths = self._skip_replay_buffer.random_next_batch(skip_batch_size)
                    test = self._q_target(batch_next_states)[torch.arange(skip_batch_size).long(), torch.argmax(
                                    self._q(batch_next_states), dim=1)]
                    skip_targets = batch_rewards + (1 - batch_terminal_flags) * np.power(self._gamma, batch_lengths) * \
                                self._q_target(batch_next_states)[torch.arange(skip_batch_size).long(), torch.argmax(
                                    self._q(batch_next_states), dim=1)]
                    skip_predictions = self._skip_q(batch_states)[torch.arange(skip_batch_size).long(), (batch_lengths-1).long()]
                    skip_loss = self._skip_loss_function(skip_predictions, skip_targets.detach())
                    self._skip_q_optimizer.zero_grad()
                    skip_loss.backward()
                    self._skip_q_optimizer.step()
                    
                    # update Q network
                    self._replay_buffer.add_transition(s, a, ns, r, done)
                    batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = \
                        self._replay_buffer.random_next_batch(batch_size)
                    target = batch_rewards + (1 - batch_terminal_flags) * self._gamma * \
                             self._q_target(batch_next_states)[torch.arange(batch_size).long(), torch.argmax(
                                 self._q(batch_next_states), dim=1)]
                    prediction = self._q(batch_states)[torch.arange(batch_size).long(), batch_actions.long()]
                    loss = self._loss_function(prediction, target.detach())
                    self._q_optimizer.zero_grad()
                    loss.backward()
                    self._q_optimizer.step()
                    soft_update(self._q_target, self._q, 0.01)

                    s = ns

                    if done or (ep_n_steps>=max_env_time_steps) or (total_steps>=max_env_time_steps):
                        break

                if done or (ep_n_steps>=max_env_time_steps) or (total_steps>=max_env_time_steps):
                        break

            if done or (ep_n_steps>=max_env_time_steps) or (total_steps>=max_env_time_steps):
                break
        
        

    def __repr__(self):
        return "TDQN"

    def save_model(self, model_name):
        torch.save(self._q.state_dict(), os.path.join(self.out_dir, model_name + ".pt"))
        torch.save(self._skip_q.state_dict(), os.path.join(self.out_dir, model_name + "_skip.pt"))

    def load(self, q_path, skip_q_path):
        self._q.load_state_dict(torch.load(q_path))
        self._skip_q.load_state_dict(torch.load(skip_q_path))


