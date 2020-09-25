"""
Code adapted from
"Dynamic Algorithm Configuration:Foundation of a New Meta-Algorithmic Framework"
by A. Biedenkapp and H. F. Bozkurt and T. Eimer and F. Hutter and M. Lindauer.
Original environment authors: André Biedenkapp, H. Furkan Bozkurt
"""

import argparse
import logging
import sys
from collections import defaultdict, namedtuple
import glob
import numpy as np
import datetime
import pickle
import os
from functools import partial
from daclib.benchmarks import LubyBenchmark


EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards", "expected_rewards"])


class QTable(dict):
    def __init__(self, n_actions, float_to_int=False, **kwargs):
        """
        Look up table for state-action values.
        :param n_actions: action space size
        :param float_to_int: flag to determine if state values need to be rounded to the closest integer
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
        return tuple(item) in self.__table.keys()

    def keys(self):
        return self.__table.keys()


def make_epsilon_greedy_policy(Q: QTable, epsilon: float, nA: int) -> callable:
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    I.e. create weight vector from which actions get sampled.
    :param Q: tabular state-action lookup function
    :param epsilon: exploration factor
    :param nA: size of action space to consider for this policy
    """

    def policy_fn(observation):
        policy = np.ones(nA) * epsilon / nA
        best_action = np.random.choice(np.argwhere(  # random choice for tie-breaking only
            Q[observation] == np.amax(Q[observation])
        ).flatten())
        policy[best_action] += (1 - epsilon)
        return policy

    return policy_fn


def get_decay_schedule(start_val: float, decay_start: int, num_episodes: int, type_: str):
    """
    Create epsilon decay schedule
    :param start_val: Start decay from this value (i.e. 1)
    :param decay_start: number of iterations to start epsilon decay after
    :param num_episodes: Total number of episodes to decay over
    :param type_: Which strategy to use. Implemented choices: 'const', 'log', 'linear'
    :return:
    """
    if type_ == 'const':
        return np.array([start_val for _ in range(num_episodes)])
    elif type_ == 'log':
        return np.hstack([[start_val for _ in range(decay_start)],
                          np.logspace(np.log10(start_val), np.log10(0.000001), (num_episodes - decay_start))])
    elif type_ == 'linear':
        return np.hstack([[start_val for _ in range(decay_start)],
                          np.linspace(start_val, 0, (num_episodes - decay_start))])
    else:
        raise NotImplementedError


def greedy_eval_Q(Q: QTable, this_environment, nevaluations: int = 1):
    """
    Evaluate Q function greediely with epsilon=0
    :returns average cumulative reward, the expected reward after resetting the environment, episode length
    """
    cumuls = []
    for _ in range(nevaluations):
        evaluation_state = this_environment.reset()
        episode_length, cummulative_reward = 0, 0
        expected_reward = np.max(Q[evaluation_state])
        greedy = make_epsilon_greedy_policy(Q, 0, this_environment.action_space.n)
        while True:  # roll out episode
            evaluation_action = np.random.choice(list(range(this_environment.action_space.n)),
                                                 p=greedy(evaluation_state))
            s_, evaluation_reward, evaluation_done, _ = this_environment.step(evaluation_action)
            cummulative_reward += evaluation_reward
            episode_length += 1
            if evaluation_done:
                break
            evaluation_state = s_
        cumuls.append(cummulative_reward)
    return np.mean(cumuls), expected_reward, episode_length  # Q, cumulative reward


def update(Q: QTable, environment, policy: callable, alpha: float, discount_factor: float):
    """
    Q update
    :param Q: state-action value look-up table
    :param environment: environment to use
    :param policy: the current policy
    :param alpha: learning rate
    :param discount_factor: discounting factor
    """
    # Need to parse to string to easily handle list as state with defdict
    policy_state = environment.reset()
    episode_length, cummulative_reward = 0, 0
    expected_reward = np.max(Q[policy_state])
    while True:  # roll out episode
        policy_action = np.random.choice(list(range(environment.action_space.n)), p=policy(policy_state))
        s_, policy_reward, policy_done, _ = environment.step(policy_action)
        cummulative_reward += policy_reward
        episode_length += 1
        Q[[policy_state, policy_action]] = Q[[policy_state, policy_action]] + alpha * (
                (policy_reward + discount_factor * Q[[s_, np.argmax(Q[s_])]]) - Q[[policy_state, policy_action]])
        if policy_done:
            break
        policy_state = s_
    return Q, cummulative_reward, expected_reward, episode_length  # Q, cumulative reward


def q_learning(environment, num_episodes: int, discount_factor: float = 1.0, alpha: float = 0.5,
               epsilon: float = 0.1, verbose: bool = False, track_test_stats: bool = False, float_state=False,
               epsilon_decay: str = 'const', decay_starts: int = 0, number_of_evaluations: int = 1,
               test_environment = None):
    """
    Q-Learning algorithm
    """
    assert 0 <= discount_factor <= 1, 'Lambda should be in [0, 1]'
    assert 0 <= epsilon <= 1, 'epsilon has to be in [0, 1]'
    assert alpha > 0, 'Learning rate has to be positive'
    # The action-value function.
    # Nested dict that maps state -> (action -> action-value).
    Q = QTable(env.action_space.n, float_state)
    test_stats = None
    if track_test_stats:
        test_stats = EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes),
            expected_rewards=np.zeros(num_episodes))

    # Keeps track of episode lengths and rewards
    train_stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        expected_rewards=np.zeros(num_episodes))

    epsilon_schedule = get_decay_schedule(epsilon, decay_starts, num_episodes, epsilon_decay)
    for i_episode in range(num_episodes):

        epsilon = epsilon_schedule[i_episode]
        # The policy we're following
        policy = make_epsilon_greedy_policy(Q, epsilon, environment.action_space.n)
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            if verbose:
                print("\rEpisode {:>5d}/{}.".format(i_episode + 1, num_episodes))
            else:
                print("\rEpisode {:>5d}/{}.".format(i_episode + 1, num_episodes), end='')
                sys.stdout.flush()
        Q, rs, exp_rew, ep_len = update(Q, environment, policy, alpha, discount_factor)
        if track_test_stats:  # Keep track of test performance, s.t. they don't influence the training Q
            if test_environment:
                test_reward, test_expected_reward, test_episode_length = greedy_eval_Q(
                    Q, test_environment, nevaluations=number_of_evaluations)
                test_stats.episode_rewards[i_episode] = test_reward
                test_stats.expected_rewards[i_episode] = test_expected_reward
                test_stats.episode_lengths[i_episode] = test_episode_length
        train_reward, train_expected_reward, train_episode_length = greedy_eval_Q(
            Q, environment, nevaluations=number_of_evaluations)
        train_stats.episode_rewards[i_episode] = train_reward
        train_stats.expected_rewards[i_episode] = train_expected_reward
        train_stats.episode_lengths[i_episode] = train_episode_length
    if not verbose:
        print("\rEpisode {:>5d}/{}.".format(i_episode + 1, num_episodes))

    return Q, (test_stats, train_stats)


def zeroOne(stringput):
    """
    Helper to keep input arguments in [0, 1]
    """
    val = float(stringput)
    if val < 0 or val > 1.:
        raise argparse.ArgumentTypeError("%r is not in [0, 1]", stringput)
    return val

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tabular Q-learning example')
    parser.add_argument('-n', '--n-eps', dest='neps',
                        default=100,
                        help='Number of episodes to roll out.',
                        type=int)
    parser.add_argument('--epsilon_decay',
                        choices=['linear', 'log', 'const'],
                        default='const',
                        help='How to decay epsilon from the given starting point to 0 or constant')
    parser.add_argument('--decay_starts',
                        type=int,
                        default=0,
                        help='How long to keep epsilon constant before decaying. '
                             'Only active if epsilon_decay log or linear')
    parser.add_argument('-r', '--repetitions',
                        default=1,
                        help='Number of repeated learning runs.',
                        type=int)
    parser.add_argument('-d', '--discount-factor', dest='df',
                        default=.99,
                        help='Discount factor',
                        type=zeroOne)
    parser.add_argument('-l', '--learning-rate', dest='lr',
                        default=.125,
                        help='Learning rate',
                        type=float)
    parser.add_argument('-e', '--epsilon',
                        default=0.1,
                        help='Epsilon for the epsilon-greedy method to follow',
                        type=zeroOne)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Use debug output')
    parser.add_argument('-o', '--out-dir', dest='out_dir',
                        default='.',
                        help='Dir to store result files in')
    parser.add_argument('--instance_feature_file', dest='inst_feats',
                        default="../instance_sets/luby_train.csv",
                        help='Instance feature file to use for sigmoid environment',
                        type=os.path.abspath)
    parser.add_argument('-s', '--seed',
                        default=0,
                        type=int)
    parser.add_argument('--cutoff',
                        default=None,
                        type=int,
                        help='Env max steps')
    parser.add_argument('--min_steps',
                        default=None,
                        type=int,
                        help='Env min steps. Only active for the lubyt* environments')
    parser.add_argument('--test_insts',
                        default=None,
                        help='Test instances to use with q-learning for evaluation purposes',
                        type=os.path.abspath)
    parser.add_argument('--reward_variance',
                        default=1.5,
                        type=float,
                        help='Variance of noisy reward signal for lubyt*')

    args = parser.parse_args()
    benchmark = LubyBenchmark()
    if args.cutoff:
        benchmark.config.cutoff = args.cutoff
    if args.min_steps:
        benchmark.config.min_steps = args.min_steps
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    np.random.seed(args.seed)
    stats, tstats = [], []
    rewards = []
    lens = []
    expects = []
    q_func = None
    if not os.path.exists(os.path.realpath(args.out_dir)):
        os.mkdir(args.out_dir)

    ds = datetime.date.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S_%f")
    random_agent = False
    if args.epsilon == 1. and args.epsilon_decay == 'const':
        folder = 'random_' + ds
        random_agent = True
    else:
        folder = 'tabular_' + ds
    folder = os.path.join(os.path.realpath(args.out_dir), folder)
    if not os.path.exists(folder):
        os.mkdir(folder)
        os.chdir(folder)
    for r in range(args.repetitions):
        logging.info('Round %d', r)
        benchmark.config.seed = args.seed
        benchmark.config.instance_set_path = args.inst_feats
        env = benchmark.get_benchmark_env()

        if args.test_insts:
            benchmark.config.seed = 2 ** 32 - 1 - args.seed
            benchmark.config.instance_set_path = args.test_insts
            test_env = benchmark.get_benchmark_env()
        else:
            test_env = None

        eval_for = 100
        float_state = False
        q_func, test_train_stats = q_learning(env, args.neps, discount_factor=args.df,
                                              alpha=args.lr, epsilon=args.epsilon,
                                              verbose=args.verbose, track_test_stats=True if test_env else False,
                                              float_state=float_state,
                                              epsilon_decay=args.epsilon_decay,
                                              decay_starts=args.decay_starts,
                                              number_of_evaluations=eval_for,
                                              test_environment=test_env)
        fn = '%04d_%s-greedy-results-luby-%d_eps-%d_reps-seed_%d.pkl' % (r, str(args.epsilon), args.neps,
                                                                       args.repetitions, args.seed)
        with open(fn.replace('results', 'q_func'), 'wb') as fh:
            pickle.dump(dict(q_func), fh)
        with open(fn.replace('results', 'stats'), 'wb') as fh:
            pickle.dump(test_train_stats[1].episode_rewards, fh)
        with open(fn.replace('results', 'lens'), 'wb') as fh:
            pickle.dump(test_train_stats[1].episode_lengths, fh)
        if args.test_insts:
            with open(fn.replace('results', 'test_stats'), 'wb') as fh:
                pickle.dump(test_train_stats[0].episode_rewards, fh)
            with open(fn.replace('results', 'test_lens'), 'wb') as fh:
                pickle.dump(test_train_stats[0].episode_lengths, fh)
        if r == args.repetitions - 1:
            logging.info('Example Q-function:')
            for i in range(10):
                print('#' * 120)
                s = env.reset()
                done = False
                while not done:
                    q_vals = q_func[s]
                    action = np.argmax(q_vals)
                    print(s, q_vals, action)
                    s, _, done, _ = env.step(action)
