from dacbench.abstract_agent import AbstractDACBenchAgent


class RandomAgent(AbstractDACBenchAgent):
    def __init__(self, env):
        self.sample_action = env.action_space.sample

    def act(self, state, reward):
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


class GenericAgent(AbstractDACBenchAgent):
    def __init__(self, env, policy):
        self.policy = policy
        self.env = env

    def act(self, state, reward):
        return self.policy(self.env, state)

    def train(self, next_state, reward):
        pass

    def end_episode(self, state, reward):
        pass
