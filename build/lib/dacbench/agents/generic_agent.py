from dacbench.abstract_agent import AbstractDACBenchAgent


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
