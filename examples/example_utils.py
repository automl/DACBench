"""Example utils."""

import argparse
from collections import defaultdict, namedtuple

import gymnasium as gym
import numpy as np

rng = np.random.default_rng()


class DummyEnv(gym.Env):
    def __init__(self):
        """Initialise the environment."""
        self.c_step = None
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(1)
        self.reward_range = (-10, 10)

    def step(self, action):
        """Make a step."""
        self.c_step += 1
        return np.array([0]), 0, False, self.c_step > 9, {}

    def reset(self):
        """Reset the environment."""
        self.c_step = 0
        return np.array([1]), {}


class QTable(dict):
    def __init__(self, n_actions, float_to_int=False, **kwargs):
        """Look up table for state-action values.
        :param n_actions: action space size
        :param float_to_int:
            flag to determine if state values need to be rounded to the closest integer
        """
        super().__init__(**kwargs)
        self.n_actions = n_actions
        self.float_to_int = float_to_int
        self.__table = defaultdict(lambda: np.zeros(n_actions))

    def __getitem__(self, item):
        try:
            table_state, table_action = item
            if self.float_to_int:
                table_state = map(int, table_state)
            return self.__table[tuple(table_state)][table_action]

        except ValueError:
            if self.float_to_int:
                item = map(int, item)
            return self.__table[tuple(item)]

    def __setitem__(self, key, value):
        try:
            table_state, table_action = key
            if self.float_to_int:
                table_state = map(int, table_state)
            self.__table[tuple(table_state)][table_action] = value
        except ValueError:
            if self.float_to_int:
                key = map(int, key)
            self.__table[tuple(key)] = value

    def __contains__(self, item):
        return tuple(item) in self.__table

    def keys(self):
        """Return the keys of the table."""
        return self.__table.keys()


def make_tabular_policy(Q: QTable, epsilon: float, nA: int) -> callable:  # noqa: N803
    """Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    I.e. create weight vector from which actions get sampled.
    :param Q: tabular state-action lookup function
    :param epsilon: exploration factor
    :param nA: size of action space to consider for this policy
    """

    def policy_fn(observation):
        policy = np.ones(nA) * epsilon / nA
        best_action = rng.choice(
            np.argwhere(  # random choice for tie-breaking only
                Q[observation] == np.amax(Q[observation])
            ).flatten()
        )
        policy[best_action] += 1 - epsilon
        return policy

    return policy_fn


def get_decay_schedule(
    start_val: float, decay_start: int, num_episodes: int, type_: str
):
    """Create epsilon decay schedule
    :param start_val: Start decay from this value (i.e. 1)
    :param decay_start: number of iterations to start epsilon decay after
    :param num_episodes: Total number of episodes to decay over
    :param type_: Which strategy to use. Implemented choices: 'const', 'log', 'linear'
    :return:
    """
    if type_ == "const":
        return np.array([start_val for _ in range(num_episodes)])

    if type_ == "log":
        return np.hstack(
            [
                [start_val for _ in range(decay_start)],
                np.logspace(
                    np.log10(start_val),
                    np.log10(0.000001),
                    (num_episodes - decay_start),
                ),
            ]
        )

    if type_ == "linear":
        return np.hstack(
            [
                [start_val for _ in range(decay_start)],
                np.linspace(start_val, 0, (num_episodes - decay_start)),
            ]
        )

    raise NotImplementedError


def greedy_eval_Q(Q: QTable, this_environment, nevaluations: int = 1):  # noqa: N802, N803
    """Evaluate Q function greediely with epsilon=0
    :returns
        average cumulative reward,
        the expected reward after resetting the environment,
        episode length
    """
    cumuls = []
    for _ in range(nevaluations):
        evaluation_state, _ = this_environment.reset()
        episode_length, cummulative_reward = 0, 0
        expected_reward = np.max(Q[evaluation_state])
        greedy = make_tabular_policy(Q, 0, this_environment.action_space.n)
        while True:  # roll out episode
            evaluation_action = rng.choice(
                list(range(this_environment.action_space.n)), p=greedy(evaluation_state)
            )
            (
                s_,
                evaluation_reward,
                eval_done,
                evaluation_done,
                _,
            ) = this_environment.step(evaluation_action)
            cummulative_reward += evaluation_reward
            episode_length += 1
            if evaluation_done or eval_done:
                break

            evaluation_state = s_
        cumuls.append(cummulative_reward)
    return np.mean(cumuls), expected_reward, episode_length  # Q, cumulative reward


def update(
    Q: QTable,  # noqa: N803
    environment,
    policy: callable,
    alpha: float,
    discount_factor: float,
):
    """Q update
    :param Q: state-action value look-up table
    :param environment: environment to use
    :param policy: the current policy
    :param alpha: learning rate
    :param discount_factor: discounting factor
    """
    # Need to parse to string to easily handle list as state with defdict
    policy_state, _ = environment.reset()
    episode_length, cummulative_reward = 0, 0
    expected_reward = np.max(Q[policy_state])
    terminated, truncated = False, False
    while not (terminated or truncated):  # roll out episode
        policy_action = rng.choice(
            list(range(environment.action_space.n)), p=policy(policy_state)
        )
        s_, policy_reward, terminated, truncated, _ = environment.step(policy_action)
        cummulative_reward += policy_reward
        episode_length += 1
        Q[[policy_state, policy_action]] = Q[[policy_state, policy_action]] + alpha * (
            (policy_reward + discount_factor * Q[[s_, np.argmax(Q[s_])]])
            - Q[[policy_state, policy_action]]
        )
        policy_state = s_
    return (
        Q,
        cummulative_reward,
        expected_reward,
        episode_length,
    )  # Q, cumulative reward


EpisodeStats = namedtuple(
    "Stats", ["episode_lengths", "episode_rewards", "expected_rewards"]
)


def zeroOne(stringput):  # noqa: N802
    """Helper to keep input arguments in [0, 1]"""
    val = float(stringput)
    if val < 0 or val > 1.0:
        raise argparse.ArgumentTypeError("%r is not in [0, 1]", stringput)

    return val
