import math

from dacbench.abstract_agent import AbstractDACBenchAgent


class CosineAnnealingAgent(AbstractDACBenchAgent):
    def __init__(self, env, base_lr=0.1, t_max=1000, eta_min=0):
        self.eta_min = eta_min
        self.t_max = t_max
        self.base_lr = base_lr
        self.current_lr = base_lr
        self.last_epoch = -1
        super(CosineAnnealingAgent, self).__init__(env)

    def act(self, state, reward):
        self.last_epoch += 1
        if self.last_epoch == 0:
            return self.base_lr
        elif (self.last_epoch - 1 - self.t_max) % (2 * self.t_max) == 0:
            return (
                self.current_lr
                + (self.base_lr - self.eta_min)
                * (1 - math.cos(math.pi / self.t_max))
                / 2
            )
        return (1 + math.cos(math.pi * self.last_epoch / self.t_max)) / (
            1 + math.cos(math.pi * (self.last_epoch - 1) / self.t_max)
        ) * (self.current_lr - self.eta_min) + self.eta_min

    def train(self, state, reward):
        pass

    def end_episode(self, state, reward):
        pass
