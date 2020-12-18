import numpy as np
from dacbench.runner import run_dacbench, plot_results
from dacbench.abstract_agent import AbstractDACBenchAgent
from example_utils import QTable, make_tabular_policy
from gym import spaces


class ExampleTabularAgent(AbstractDACBenchAgent):
    def __init__(self, env, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        if isinstance(env.action_space, spaces.Discrete):
            self.num_actions = env.action_space.n
        else:
            self.num_actions = int(env.action_space.high[0])
        self.q = QTable(self.num_actions)
        self.policy = make_tabular_policy(self.q, self.epsilon, self.num_actions)
        self.action = None
        self.state = None

    def act(self, state, reward):
        self.action = np.random.choice(
            list(range(self.num_actions)), p=self.policy(state)
        )
        self.state = state
        return self.action

    def train(self, next_state, reward):
        self.q[[self.state, self.action]] = self.q[
            [self.state, self.action]
        ] + self.alpha * (
            (reward + self.gamma * self.q[[next_state, np.argmax(self.q[next_state])]])
            - self.q[[self.state, self.action]]
        )

    def end_episode(self, state, reward):
        pass


def make_agent(env):
    return ExampleTabularAgent(env)


path = "dacbench_tabular"
run_dacbench(path, make_agent, 10)
plot_results(path)
