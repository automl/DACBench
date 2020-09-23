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

# Instance IDEA 1: shift luby seq -> feat is sum of skipped action values
# Instance IDEA 2: "Wiggle" luby i.e. luby(t + N(0, 0.1)) -> feat is sampled value
class LubyEnv(AbstractEnv):
    """
    Luby "cyclic" benchmark
    """

    def __init__(self,
                 config
                 #min_steps: int=2**3,
                 #max_steps: int=2**6,
                 #seed: int=0,
                 #fuzzy: bool=False,
                 #instance_mode: int=0,
                 #instance_feats: str=None,
                 #noise_sig: float=1.5
                 ) -> None:
        super().__init__(config)
        self.rng = np.random.RandomState(seed)
        self._c_step = 0
        self.logger = None

        self._hist_len = config["hist_length"]
        self._ms = self.n_steps
        self._mi = config['min_steps']
        self._state = np.array([-1 for _ in range(self._hist_len + 1)])
        self._r = 0
        self._genny = luby_gen(1)
        self._next_goal = next(self._genny)
        # Generate luby sequence up to 2*max_steps + 2 as mode 1 could potentially shift up to max_steps
        self.__seq = np.log2([next(luby_gen(i)) for i in range(1, 2*max_steps + 2)])
        self._jenny_i = 1
        self._fuzz = config['fuzzy']
        #self.observation_space = spaces.Box(
        #    low=np.array([-1 for _ in range(self._hist_len + additional_feats)]),
        #    high=np.array([2**max(self.__seq + 1) for _ in range(self._hist_len + additional_feats)]),
        #    dtype=np.float32)
        self.logger = logging.getLogger(self.__str__())

        self._start_dist = None
        self._sticky_dis = None
        self._sticky_shif = 0
        self._start_shift = 0
        self.__lower, self.__upper = 0, 0
        self.__error = 0

    def step(self, action: int):
        """Function to interact with the environment.
            Args:
            action (int): one of [1, 2, 4, 8, 16, 32, 64, 128]/[0, 1, 2, 3, 4, 5, 6, 7]
        Returns:
            next_state (List[int]):  Next state observed from the environment.
            reward (float):
            done (bool):  Specifies if environment is solved.
            info (None):
        """
        done = super().reset()
        prev_state = self._state.copy()
        if action == self._next_goal:
            self._r = 0  # we don't want to allow for exploiting large rewards by tending towards long sequences
        else:  # mean and var chosen s.t. ~1/4 of rewards are positive
            self._r = -1 if not self._fuzz else self.rng.normal(-1, self.noise_sig)

        if self.__error < self.__lower:  # needed to avoid too long sequences of sticky actions
            self.__error += np.abs(self.__lower)
        elif self.__error > self.__upper:
            self.__error -= np.abs(self.__upper)
        self._jenny_i += 1
        self.__error += self._sticky_shif

        # next target in sequence at step luby_t is determined by the current time step (jenny_i), the start_shift
        # value and the sticky error. Additive sticky error leads to sometimes rounding to the next time_step and
        # thereby repeated actions. With check against lower/upper we reset the sequence to the correct timestep in
        # the t+1 timestep.
        luby_t = max(1, int(np.round(self._jenny_i + self._start_shift + self.__error)))
        self._next_goal = self.__seq[luby_t - 1]
        if self._c_step - 1 < self._hist_len:
            self._state[(self._c_step-1)] = action
        else:
            self._state[:-self.__n_feats - 1] = self._state[1:-self.__n_feats]
            self._state[-self.__n_feats - 1] = action
        self._state[-self.__n_feats] = self._c_step - 1
        next_state = self._state if not done else prev_state
        self.logger.debug("i: (s, a, r, s') / %+5d: (%s, %d, %5.2f, %2s)     g: %3d  l: %3d", self._c_step-1,
                          str(prev_state),
                          action, self._r, str(next_state),
                          int(self._next_goal), self.n_steps)
        return np.array(next_state), self._r, done, {}

    def reset(self) -> List[int]:
        """
          Returns:
            next_state (int):  Next state observed from the environment.
        """
        super.reset()
        self._r = 0
        self.n_steps = self._mi

        self.__error = 0 + self._sticky_shif
        self._jenny_i = 1
        luby_t = max(1, int(np.round(self._jenny_i + self._start_shift + self.__error)))
        self._next_goal = self.__seq[luby_t - 1]
        self.logger.debug("i: (s, a, r, s') / %+5d: (%2d, %d, %5.2f, %2d)     g: %3d  l: %3d", -1, -1, -1, -1, -1,
                          int(self._next_goal), self.n_steps)
        self._state = [-1 for _ in range(self._hist_len + self.__n_feats)]
        return np.array(self._state)

    def close(self) -> bool:
        return True

    def render(self, mode: str='human', close: bool=True) -> None:
        if mode != 'human':
            raise NotImplementedError
        pass

def luby_gen(i):
    for k in range(1, 33):
        if i == ((1 << k) - 1):
            yield 1 << (k-1)
    for k in range(1, 9999):
        if 1 << (k - 1) <= i < (1 << k) - 1:
            for x in luby_gen(i - (1 << (k-1)) + 1):
                yield x
