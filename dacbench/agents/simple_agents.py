from gymnasium import spaces

from dacbench.abstract_agent import AbstractDACBenchAgent


class RandomAgent(AbstractDACBenchAgent):
    def __init__(self, env):
        self.sample_action = env.action_space.sample
        self.shortbox = isinstance(env.action_space, spaces.Box)
        if self.shortbox:
            self.shortbox = self.shortbox and len(env.action_space.low) == 1

    def act(self, state, reward):
        if self.shortbox:
            return self.sample_action()[0]
        else:
            return self.sample_action()

    def train(self, next_state, reward):
        pass

    def end_episode(self, state, reward):
        pass


class StaticAgent(AbstractDACBenchAgent):
    def __init__(self, env, action):
        self.action = action

    def act(self, state, reward):
        return self.action

    def train(self, next_state, reward):
        pass

    def end_episode(self, state, reward):
        pass
