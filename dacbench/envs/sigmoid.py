"""
Sigmoid environment from
"Dynamic Algorithm Configuration:Foundation of a New Meta-Algorithmic Framework"
by A. Biedenkapp and H. F. Bozkurt and T. Eimer and F. Hutter and M. Lindauer.
Original environment authors: André Biedenkapp, H. Furkan Bozkurt
"""

import itertools
from typing import List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from dacbench import AbstractEnv


class SigmoidEnv(AbstractEnv):
    """
    Environment for tracing sigmoid curves
    """

    def _sig(self, x, scaling, inflection):
        """ Simple sigmoid function """
        return 1 / (1 + np.exp(-scaling * (x - inflection)))

    def __init__(self, config) -> None:
        """
        Initialize Sigmoid Env

        Parameters
        -------
        config : objdict
            Environment configuration
        """
        super(SigmoidEnv, self).__init__(config)

        self.shifts = [self.n_steps / 2 for _ in config["action_values"]]
        self.slopes = [-1 for _ in config["action_values"]]
        self.slope_multiplier = config["slope_multiplier"]
        self.action_vals = config["action_values"]
        self.n_actions = len(self.action_vals)
        self.action_mapper = {}
        for idx, prod_idx in zip(
            range(np.prod(config["action_values"])),
            itertools.product(*[np.arange(val) for val in config["action_values"]]),
        ):
            self.action_mapper[idx] = prod_idx
        self._prev_state = None
        self.action = None

        if "reward_function" in config.keys():
            self.get_reward = config["reward_function"]
        else:
            self.get_reward = self.get_default_reward

        if "state_method" in config.keys():
            self.get_state = config["state_method"]
        else:
            self.get_state = self.get_default_state

    def step(self, action: int):
        """
        Execute environment step

        Parameters
        ----------
        action : int
            action to execute

        Returns
        -------
        np.array, float, bool, dict
            state, reward, done, info
        """
        self.done = super(SigmoidEnv, self).step_()
        action = self.action_mapper[action]
        assert self.n_actions == len(
            action
        ), f"action should be of length {self.n_actions}."

        self.action = action
        next_state = self.get_state(self)
        self._prev_state = next_state
        return next_state, self.get_reward(self), self.done, {}

    def reset(self) -> List[int]:
        """
        Resets env

        Returns
        -------
        numpy.array
            Environment state
        """
        super(SigmoidEnv, self).reset_()
        self.shifts = self.instance[: self.n_actions]
        self.slopes = self.instance[self.n_actions :]
        self._prev_state = None
        return self.get_state(self)

    def get_default_reward(self, _):
        r = [
            1 - np.abs(self._sig(self.c_step, slope, shift) - (act / (max_act - 1)))
            for slope, shift, act, max_act in zip(
                self.slopes, self.shifts, self.action, self.action_vals
            )
        ]
        r = np.prod(r)
        r = max(self.reward_range[0], min(self.reward_range[1], r))
        return r

    def get_default_state(self, _):
        remaining_budget = self.n_steps - self.c_step
        next_state = [remaining_budget]
        for shift, slope in zip(self.shifts, self.slopes):
            next_state.append(shift)
            next_state.append(slope)
        if self.c_step == 0:
            next_state += [-1 for _ in range(self.n_actions)]
        else:
            next_state += self.action
        return np.array(next_state)

    def close(self) -> bool:
        """
        Close Env

        Returns
        -------
        bool
            Closing confirmation
        """
        return True

    def render(self, mode: str) -> None:
        """
        Render env in human mode

        Parameters
        ----------
        mode : str
            Execution mode
        """
        if mode == "human" and self.n_actions == 2:
            plt.ion()
            plt.show()
            plt.cla()
            steps = np.arange(self.n_steps)
            self.data = self._sig(steps, self.slopes[0], self.shifts[0]) * self._sig(
                steps, self.slopes[1], self.shifts[1]
            ).reshape(-1, 1)

            plt.imshow(
                self.data,
                extent=(0, self.n_steps - 1, 0, self.n_steps - 1),
                interpolation="nearest",
                cmap=cm.plasma,
            )
            plt.axvline(x=self.c_step, color="r", linestyle="-", linewidth=2)
            plt.axhline(y=self.c_step, color="r", linestyle="-", linewidth=2)

            plt.draw()
            plt.pause(0.005)


class ContinuousStateSigmoidEnv(SigmoidEnv):
    """
    Environment for tracing sigmoid curves with a continuous state on the x-axis
    """

    def __init__(self, config) -> None:
        """
        Initialize Sigmoid Env

        Parameters
        -------
        config : objdict
            Environment configuration
        """
        super().__init__(config)

    def step(self, action: int):
        """
        Execute environment step

        Parameters
        ----------
        action : int
            action to execute

        Returns
        -------
        np.array, float, bool, dict
            state, reward, done, info
        """
        action = self.action_mapper[action]
        assert self.n_actions == len(
            action
        ), f"action should be of length {self.n_actions}."

        self.action = action
        # The reward measures how wrong the choice was so we can take this error to determine how far we travel along
        # the x-axis instead of always advancing + 1
        r = self.get_reward(self)

        # magic constants but such that the max step is ~1 and the min step is ~0.25
        self.c_step += (r + np.sqrt(np.power(r, 2) + 0.25))/2

        if self.c_step >= self.n_steps:
            self.done = True
        else:
            self.done = False

        # self.c_step is used in get_next_state to show how much distance along the x-axis is left to cover
        # Thus we get a continuous state this way.
        next_state = self.get_state(self)
        self._prev_state = next_state
        return next_state, r, self.done, {}

class ContinuousSigmoidEnv(SigmoidEnv):
    """
    Environment for tracing sigmoid curves with a continuous state on the x-axis
    """

    def __init__(self, config) -> None:
        """
        Initialize Sigmoid Env

        Parameters
        -------
        config : objdict
            Environment configuration
        """
        super().__init__(config)

    def step(self, action: np.ndarray):
        """
        Execute environment step. !!NOTE!! The action here is a list of floats and not a single number !!NOTE!!

        Parameters
        ----------
        action : list of floats
            action(s) to execute

        Returns
        -------
        np.array, float, bool, dict
            state, reward, done, info
        """
        assert self.n_actions == len(
            action
        ), f"action should be of length {self.n_actions}."

        self.action = action
        # The reward measures how wrong the choice was so we can take this error to determine how far we travel along
        # the x-axis instead of always advancing + 1
        r = self.get_reward(self)

        # magic constants but such that the max step is ~1 and the min step is ~0.25
        self.c_step += (r + np.sqrt(np.power(r, 2) + 0.25)) / 2

        if self.c_step >= self.n_steps:
            self.done = True
        else:
            self.done = False

        # self.c_step is used in get_next_state to show how much distance along the x-axis is left to cover
        # Thus we get a continuous state this way.
        next_state = self.get_state(self)
        self._prev_state = next_state
        return next_state, r, self.done, {}


if __name__ == '__main__':
    from dacbench.abstract_benchmark import objdict
    config = objdict(
        {
            "action_space_class": "Box",
            "action_space_args": [
                np.array([-np.inf for _ in range(1 + 2 * 3)]),
                np.array([np.inf for _ in range(1 + 2 * 3)]),
            ],
            "observation_space_class": "Box",
            "observation_space_type": np.float32,
            "observation_space_args": [
                np.array([-np.inf for _ in range(1 + 2 * 3)]),
                np.array([np.inf for _ in range(1 + 2 * 3)]),
            ],
            "reward_range": (0, 1),
            "cutoff": 10,
            "action_values": (2, 2),
            "slope_multiplier": 2.0,
            "seed": 0,
            "instance_set_path": "../instance_sets/sigmoid/sigmoid_2D3M_train.csv",
            "benchmark_info": None,
            'instance_set': {0:[5.847291747472278,6.063505157165379,5.356361033331866,8.473324526654427],
                             1:[5.699459023308639,0.17993881762205755,3.4218338308013356,8.486280024502191],
                             2:[5.410536230957515,5.700091608324946,-5.3540400976249165,2.76787147719077],
                             3:[1.5799464875295817,6.374885201056433,1.0378986341827443,4.219330699379608],
                             4:[2.61235568666599,6.478051235772757,7.622760392199338,-3.0898869570275167]},
        }
    )
    env = ContinuousSigmoidEnv(config)
    done = False
    s = env.reset()
    env.render(mode='human')
    while not done:
        a = [np.random.rand(), np.random.rand()]
        print(env.c_step, a)
        s, r, done, _ = env.step(a)
        env.render('human')


    config['action_space'] = "Discrete"
    config["action_space_args"] =  [int(np.prod((2, 2)))],
    env = ContinuousStateSigmoidEnv(config)
    done = False
    s = env.reset()
    env.render(mode='human')
    while not done:
        a = np.random.randint(4)
        print(env.c_step, a)
        s, r, done, _ = env.step(a)
        env.render('human')
