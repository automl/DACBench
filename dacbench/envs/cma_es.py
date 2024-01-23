import re

import numpy as np
from IOHexperimenter import IOH_function
from modcma import ModularCMAES, Parameters

from dacbench import AbstractMADACEnv


class CMAESEnv(AbstractMADACEnv):
    def __init__(self, config):
        super().__init__(config)

        self.es = None
        self.budget = config.budget
        self.total_budget = self.budget

        # Find all set hyperparam_defaults and replace cma defaults
        if "config_space" in config.keys():
            for name in config["config_space"].keys():
                value = self.config.get(name)
                if value:
                    self.representation_dict[self.uniform_name(name)] = value

        self.get_reward = config.get("reward_function", self.get_default_reward)
        self.get_state = config.get("state_method", self.get_default_state)

    def uniform_name(self, name):
        # Convert name of parameters uniformly to lowercase, separated with _ and no numbers
        pattern = r"^\d+_"

        # Use re.sub to remove the leading number and underscore
        result = re.sub(pattern, "", name)
        return result.lower()

    def reset(self, seed=None, options={}):
        super().reset_(seed)
        self.dim, self.fid, self.iid, self.representation = self.instance
        self.representation_dict = {
            "active": self.representation[0],
            "elitist": self.representation[1],
            "orthogonal": self.representation[2],
            "sequential": self.representation[3],
            "threshold_convergence": self.representation[4],
            "step_size_adaptation": self.representation[5],
            "mirrored": self.representation[6],
            "base_sampler": self.representation[7],
            "weights_option": self.representation[8],
            "local_restart": self.representation[9],
            "bound_correction": self.representation[10],
        }
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
        if isinstance(action, dict):
            for hp in action.keys():
                n_name = self.uniform_name(hp)
                if n_name == "step_size":
                    # Step size is set separately
                    self.es.parameters.sigma = action[hp][0]
                else:
                    # Save parameter values from actions
                    complete_action[n_name] = action[hp]

            # Complete the given action with defaults
            for default in self.representation_dict.keys():
                if default == "step_size":
                    continue
                if default not in complete_action:
                    complete_action[default] = self.representation_dict[default]
            complete_action = complete_action.values()
        else:
            raise ValueError("Action must be a Dict")

        new_parameters = Parameters.from_config_array(self.dim, complete_action)
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

    def render(self, mode="human"):
        raise NotImplementedError("CMA-ES does not support rendering at this point")
