import itertools
import logging
import os
import sys
from typing import List, Tuple

import gym
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from gym import Env, spaces, wrappers
from scipy.stats import truncnorm

import AbstractEnv

class Sigmoid(AbstractEnv):
    """
    Sigmoid reward
    """

    def _sig(self, x, scaling, inflection):
        """ Simple sigmoid """
        return 1 / (1 + np.exp(-scaling * (x - inflection)))

    def __init__(self,
                 config,
                 #n_actions: int=2,
                 #action_vals: tuple=(5, 10),
                 #seed: bool=0,
                 #noise: float=0.0,
                 #instance_feats: str=None,
                 #slope_multiplier: float=2
                 ) -> None:
        super().__init__()
        #self.n_actions = n_actions
        self.rng = np.random.RandomState(seed)
        self._c_step = 0
        #assert self.n_actions == len(action_vals), (
        #    f'action_vals should be of length {self.n_actions}.')
        self.shifts = [self.n_steps / 2 for _ in config['action_vals']]
        self.slopes = [-1 for _ in config['action_vals']]
        #self.reward_range = (0, 1)
        self.slope_multiplier = slope_multiplier
        self.action_vals = action_vals
        # budget spent, inst_feat_1, inst_feat_2
        # self._state = [-1 for _ in range(3)]
        # self.action_space = spaces.MultiDiscrete(action_vals)
        #self.action_space = spaces.Discrete(int(np.prod(action_vals)))
        self.action_mapper = {}
        for idx, prod_idx in zip(range(np.prod(config['action_vals'])),
                                       itertools.product(*[np.arange(val) for val in action_vals])):
            self.action_mapper[idx] = prod_idx
        self.observation_space = spaces.Box(
            low=np.array([-np.inf for _ in range(1 + self.n_actions * 3)]),
            high=np.array([np.inf for _ in range(1 + self.n_actions * 3)]))
        self.logger = logging.getLogger(self.__str__())
        self._prev_state = None

    def step(self, action: int):
        done = super().step_()
        action = self.action_mapper[action]
        assert self.n_actions == len(action), (
            f'action should be of length {self.n_actions}.')

        val = self._c_step
        r = [1 - np.abs(self._sig(val, slope, shift) - (act / (max_act - 1)))
            for slope, shift, act, max_act in zip(
                self.slopes, self.shifts, action, self.action_vals
            )]
        r = np.clip(np.prod(r), 0.0, 1.0)
        remaining_budget = self.n_steps - self._c_step

        next_state = [remaining_budget]
        for shift, slope in zip(self.shifts, self.slopes):
            next_state.append(shift)
            next_state.append(slope)
        next_state += action
        prev_state = self._prev_state

        self.logger.debug("i: (s, a, r, s') / %d: (%s, %d, %5.2f, %2s)", self._c_step-1, str(prev_state),
                          action, r, str(next_state))
        self._c_step += 1
        self._prev_state = next_state
        return np.array(next_state), r, self._c_step >= self.n_steps, {}

    def reset(self) -> List[int]:
        super.reset_()
        remaining_budget = self.n_steps - self._c_step
        next_state = [remaining_budget]
        for shift, slope in zip(self.shifts, self.slopes):
            next_state.append(shift)
            next_state.append(slope)
        next_state += [-1 for _ in range(self.n_actions)]
        self._prev_state = None
        self.logger.debug("i: (s, a, r, s') / %d: (%2d, %d, %5.2f, %2d)", -1, -1, -1, -1, -1)
        return np.array(next_state)

    def close(self) -> bool:
        return True

    def render(self, mode: str, close: bool=True) -> None:
        if mode == 'human' and self.n_actions == 2:
            plt.ion()
            plt.show()
            plt.cla()
            steps = np.arange(self.n_steps)
            self.data = self._sig(steps, self.slopes[0], self.shifts[0]) * \
                        self._sig(steps, self.slopes[1], self.shifts[1]).reshape(-1, 1)

            plt.imshow(
                self.data,
                extent=(0, self.n_steps - 1, 0, self.n_steps - 1),
                interpolation='nearest', cmap=cm.plasma)
            plt.axvline(x=self._c_step, color='r', linestyle='-', linewidth=2)
            plt.axhline(y=self._c_step, color='r', linestyle='-', linewidth=2)

            plt.draw()
            plt.pause(0.005)
