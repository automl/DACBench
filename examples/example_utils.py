import gym
import argparse
import numpy as np
from collections import defaultdict, namedtuple
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
from chainerrl import q_functions, replay_buffer, explorers, policies, links, v_function
from chainerrl.agents import DQN, a3c
from chainerrl.recurrent import RecurrentChainMixin

from dacbench.logger import Logger


class DummyEnv(gym.Env):
    def __init__(self):
        self.c_step = None
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(1)
        self.reward_range = (-10, 10)

    def step(self, action):
        self.c_step += 1
        return np.array([0]), 0, self.c_step > 9, {}

    def reset(self):
        self.c_step = 0
        return np.array([1])


class QTable(dict):
    def __init__(self, n_actions, float_to_int=False, **kwargs):
        """
        Look up table for state-action values.
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
        return tuple(item) in self.__table.keys()

    def keys(self):
        return self.__table.keys()


def make_tabular_policy(Q: QTable, epsilon: float, nA: int) -> callable:
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    I.e. create weight vector from which actions get sampled.
    :param Q: tabular state-action lookup function
    :param epsilon: exploration factor
    :param nA: size of action space to consider for this policy
    """

    def policy_fn(observation):
        policy = np.ones(nA) * epsilon / nA
        best_action = np.random.choice(
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
    """
    Create epsilon decay schedule
    :param start_val: Start decay from this value (i.e. 1)
    :param decay_start: number of iterations to start epsilon decay after
    :param num_episodes: Total number of episodes to decay over
    :param type_: Which strategy to use. Implemented choices: 'const', 'log', 'linear'
    :return:
    """
    if type_ == "const":
        return np.array([start_val for _ in range(num_episodes)])

    elif type_ == "log":
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

    elif type_ == "linear":
        return np.hstack(
            [
                [start_val for _ in range(decay_start)],
                np.linspace(start_val, 0, (num_episodes - decay_start)),
            ]
        )

    else:
        raise NotImplementedError


def greedy_eval_Q(Q: QTable, this_environment, nevaluations: int = 1):
    """
    Evaluate Q function greediely with epsilon=0
    :returns
        average cumulative reward,
        the expected reward after resetting the environment,
        episode length
    """
    cumuls = []
    for _ in range(nevaluations):
        evaluation_state = this_environment.reset()
        episode_length, cummulative_reward = 0, 0
        expected_reward = np.max(Q[evaluation_state])
        greedy = make_tabular_policy(Q, 0, this_environment.action_space.n)
        while True:  # roll out episode
            evaluation_action = np.random.choice(
                list(range(this_environment.action_space.n)), p=greedy(evaluation_state)
            )
            s_, evaluation_reward, evaluation_done, _ = this_environment.step(
                evaluation_action
            )
            cummulative_reward += evaluation_reward
            episode_length += 1
            if evaluation_done:
                break

            evaluation_state = s_
        cumuls.append(cummulative_reward)
    return np.mean(cumuls), expected_reward, episode_length  # Q, cumulative reward


def update(
    Q: QTable, environment, policy: callable, alpha: float, discount_factor: float
):
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
    done = False
    while not done:  # roll out episode
        policy_action = np.random.choice(
            list(range(environment.action_space.n)), p=policy(policy_state)
        )
        s_, policy_reward, done, _ = environment.step(policy_action)
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


def zeroOne(stringput):
    """
    Helper to keep input arguments in [0, 1]
    """
    val = float(stringput)
    if val < 0 or val > 1.0:
        raise argparse.ArgumentTypeError("%r is not in [0, 1]", stringput)

    return val


# Example model class taken from chainerrl examples:
# https://github.com/chainer/chainerrl/blob/master/examples/gym/train_a3c_gym.py


class A3CFFSoftmax(chainer.ChainList, a3c.A3CModel):
    """An example of A3C feedforward softmax policy."""

    def __init__(self, ndim_obs, n_actions, hidden_sizes=(200, 200)):
        self.pi = policies.SoftmaxPolicy(
            model=links.MLP(ndim_obs, n_actions, hidden_sizes)
        )
        self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


class A3CLSTMGaussian(chainer.ChainList, a3c.A3CModel, RecurrentChainMixin):
    """An example of A3C recurrent Gaussian policy."""

    def __init__(self, obs_size, action_size, hidden_size=200, lstm_size=128):
        self.pi_head = L.Linear(obs_size, hidden_size)
        self.v_head = L.Linear(obs_size, hidden_size)
        self.pi_lstm = L.LSTM(hidden_size, lstm_size)
        self.v_lstm = L.LSTM(hidden_size, lstm_size)
        self.pi = policies.FCGaussianPolicy(lstm_size, action_size)
        self.v = v_function.FCVFunction(lstm_size)
        super().__init__(
            self.pi_head, self.v_head, self.pi_lstm, self.v_lstm, self.pi, self.v
        )

    def pi_and_v(self, state):
        def forward(head, lstm, tail):
            h = F.relu(head(state))
            h = lstm(h)
            return tail(h)

        pout = forward(self.pi_head, self.pi_lstm, self.pi)
        vout = forward(self.v_head, self.v_lstm, self.v)

        return pout, vout


def make_chainer_a3c(obs_size, action_size):
    model = A3CLSTMGaussian(obs_size, action_size)
    opt = optimizers.Adam(eps=1e-2)
    opt.setup(model)
    agent = a3c.A3C(model, opt, 10 ** 5, 0.9)
    return agent


def make_chainer_dqn(obs_size, action_space):
    q_func = q_functions.FCStateQFunctionWithDiscreteAction(
        obs_size, action_space.n, 50, 1
    )
    explorer = explorers.ConstantEpsilonGreedy(0.1, action_space.sample)
    opt = optimizers.Adam(eps=1e-2)
    opt.setup(q_func)
    rbuf = replay_buffer.ReplayBuffer(10 ** 5)
    agent = DQN(q_func, opt, rbuf, explorer=explorer, gamma=0.9)
    return agent


def flatten(li):
    return [value for sublist in li for value in sublist]


def train_chainer(
    agent, env, num_episodes=10, flatten_state=False, logger: Logger = None
):
    for i in range(num_episodes):
        state = env.reset()
        if flatten_state:
            state = np.array(flatten([state[k] for k in state.keys()]))
            state = state.astype(np.float32)
        done = False
        r = 0
        reward = 0
        while not done:
            action = agent.act_and_train(state, reward)
            next_state, reward, done, _ = env.step(action)
            r += reward
            if flatten_state:
                state = np.array(flatten([next_state[k] for k in next_state.keys()]))
                state = state.astype(np.float32)
            else:
                state = next_state
            if logger is not None:
                logger.next_step()
        agent.stop_episode_and_train(state, reward, done=done)
        if logger is not None:
            logger.next_episode()
        print(
            f"Episode {i}/{num_episodes}...........................................Reward: {r}"
        )
