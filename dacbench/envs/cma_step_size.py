import numpy as np
from IOHexperimenter import IOH_function
from modcma import ModularCMAES, Parameters

from dacbench import AbstractEnv


class CMAStepSizeEnv(AbstractEnv):
    def __init__(self, config):
        super().__init__(config)

        self.es = None
        self.budget = config.budget
        self.total_budget = self.budget

        self.get_reward = config.get("reward_function", self.get_default_reward)
        self.get_state = config.get("state_method", self.get_default_state)

    def reset(self, seed=None, options={}):
        super().reset_(seed)
        self.dim, self.fid, self.iid, self.representation = self.instance
        self.objective = IOH_function(
            self.fid, self.dim, self.iid, target_precision=1e-8
        )
        self.es = ModularCMAES(
            self.objective,
            parameters=Parameters.from_config_array(self.dim, self.representation),
        )
        return self.get_state(self), {}

    def step(self, action):
        truncated = super().step_()
        self.es.parameters.sigma = action
        terminated = not self.es.step()
        return self.get_state(self), self.get_reward(self), terminated, truncated, {}

    def close(self):
        return True

    def get_default_reward(self, *_):
        return max(
            self.reward_range[0], min(self.reward_range[1], -self.es.parameters.fopt)
        )

    def get_default_state(self, *_):
        return np.array(
            [
                self.es.parameters.lambda_,
                self.es.parameters.sigma,
                self.budget - self.es.parameters.used_budget,
                self.fid,
                self.iid,
            ]
        )
