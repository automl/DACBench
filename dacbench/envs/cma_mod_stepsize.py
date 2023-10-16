import numpy as np
import re
from IOHexperimenter import IOH_function
from modcma import ModularCMAES, Parameters

from dacbench import AbstractMADACEnv


class ModStepSizeCMAEnv(AbstractMADACEnv):
    def __init__(self, config):
        super().__init__(config)

        self.es = None
        self.budget = config.budget
        self.total_budget = self.budget

        param = Parameters(
            10
        )  # Dummy dim since current parameters don't rely on dimension

        # Get all defaults from modcma as dictionary
        self.hyperparam_defaults = dict(
            map(
                lambda m: (m, getattr(param, m)),
                param.__modules__,
            )
        )

        # Find all set hyperparam_defaults and replace cma defaults
        for name in config["config_space"].keys():
            value = self.config.get(name)
            if value:
                self.hyperparam_defaults[self.uniform_name(name)] = value

        self.get_reward = config.get("reward_function", self.get_default_reward)
        self.get_state = config.get("state_method", self.get_default_state)

    def uniform_name(self, name):
        # Convert name of parameters uniformly to lowercase, separated with _ and no numbers
        pattern = "\d*_(?P<name>[a-zA-Z]*)"

        uni_name = re.finditer(pattern, name)
        n_name: str = ""
        for n in uni_name:
            if n_name != "":
                n_name += "_"
            n_name += n.group("name")
        n_name = n_name.lower()
        return n_name

    def reset(self, seed=None, options={}):
        super().reset_(seed)
        self.dim, self.fid, self.iid, self.representation = self.instance
        self.objective = IOH_function(
            self.fid, self.dim, self.iid, target_precision=1e-8
        )
        self.es = ModularCMAES(
            self.objective,
            parameters=Parameters.from_config_array(
                self.dim, np.array(self.representation).astype(int)
            ),
        )
        return self.get_state(self), {}

    def step(self, action):
        truncated = super().step_()

        # Get all action values and uniform names
        complete_action = {}
        for hp in action.keys():
            n_name = self.uniform_name(hp)
            if n_name == "step_size":
                # Step size is set separately
                self.es.parameters.sigma = action[hp][0]
            else:
                # Save parameter values from actions
                complete_action[n_name] = action[hp]

        # Complete the given action with defaults
        for default in self.hyperparam_defaults.keys():
            if default == "step_size":
                continue
            if default not in complete_action:
                complete_action[default] = self.hyperparam_defaults[default]

        new_parameters = Parameters.from_config_array(
            self.dim, complete_action.values()
        )
        self.es.parameters.update(
            {m: getattr(new_parameters, m) for m in Parameters.__modules__}
        )

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
