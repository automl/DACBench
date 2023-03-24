from gymnasium import spaces

from dacbench.abstract_agent import AbstractDACBenchAgent


class DynamicRandomAgent(AbstractDACBenchAgent):
    def __init__(self, env, switching_interval):
        self.sample_action = env.action_space.sample
        self.switching_interval = switching_interval
        self.count = 0
        self.action = self.sample_action()
        self.shortbox = (
            isinstance(env.action_space, spaces.Box) and len(env.action_space.low) == 1
        )

    def act(self, state, reward):
        if self.count >= self.switching_interval:
            self.action = self.sample_action()
            self.count = 0
        self.count += 1

        if self.shortbox:
            return self.action[0]
        else:
            return self.action

    def train(self, next_state, reward):
        pass

    def end_episode(self, state, reward):
        pass
