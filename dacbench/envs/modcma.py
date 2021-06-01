import numpy as np
from modcma import ModularCMAES, Parameters
from IOHexperimenter import IOH_function

from dacbench import AbstractEnv


class ModCMAEnv(AbstractEnv):
    def __init__(self, config):
        super().__init__(config)

        self.es = None
        self.budget = config.budget
        self.total_budget = self.budget

        self.get_reward = config.get("reward_function", self.get_default_reward)
        self.get_state = config.get("state_method", self.get_default_state)

    def reset(self):
        super().reset_()
        self.dim, self.fid, self.iid, self.representation = self.instance
        self.objective = IOH_function(
            self.fid, self.dim, self.iid, target_precision=1e-8
        )
        self.es = ModularCMAES(
            self.objective,
            parameters=Parameters.from_config_array(
                self.dim, self.representation.astype(int)
            ),
        )
        return self.get_state(self)

    def step(self, action):
        done = super().step_()
        new_parameters = Parameters.from_config_array(self.dim, action)
        self.es.parameters.update(
            {m: getattr(new_parameters, m) for m in Parameters.__modules__}
        )
        done = not self.es.step()
        return self.get_state(self), self.get_reward(self), done, {}

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
