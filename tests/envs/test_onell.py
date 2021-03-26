import unittest

import numpy as np
from dacbench import AbstractEnv
from dacbench.envs import OneLLEnv
from dacbench.additional_configs.onell.configs import (
    onell_lbd_theory,
    onell_lbd_onefifth,
    onell_lbd_p_c,
    onell_lbd1_lbd2_p_c,
)
from dacbench.abstract_benchmark import objdict

onell_configs = [
    onell_lbd_theory,
    onell_lbd_onefifth,
    onell_lbd_p_c,
    onell_lbd1_lbd2_p_c,
]


class TestOneLLEnv(unittest.TestCase):
    def make_env(self, config):
        config["instance_set"] = {0: objdict({"size": 2000, "max_evals": 30000})}
        env = OneLLEnv(config)
        return env

    def test_setup(self):
        for i, config in enumerate(onell_configs):
            env = self.make_env(config)
            self.assertTrue(issubclass(type(env), AbstractEnv))
            self.assertFalse(env.np_random is None)

            for var_name in ["include_xprime", "count_different_inds_only"]:
                self.assertTrue(vars(env)[var_name] == config[var_name])
            print(globals()["onell_configs"][i])
            self.assertTrue(
                env.problem.__name__ == globals()["onell_configs"][i]["problem"]
            )

            self.assertTrue(len(env.state_var_names) == len(env.state_functions))

    def test_reset(self):
        for config in onell_configs:
            env = self.make_env(config)
            env.reset()
            self.assertFalse(env.n is None)
            self.assertFalse(env.max_evals is None)
            self.assertFalse(env.x is None)
            self.assertTrue(env.total_evals == 1)

    def test_get_state(self):
        # basic tests
        for config in onell_configs:
            env = self.make_env(config)
            state = env.reset()
            self.assertTrue(issubclass(type(state), np.ndarray))
            self.assertTrue(len(env.state_var_names) == len(state))

        # test if histories are updated and retrieved correctly
        env = self.make_env(onell_lbd_onefifth)
        state = env.reset()
        for i in range(10):
            state, reward, done, _ = env.step(
                np.random.choice(np.arange(10, 20), size=1)
            )
        self.assertTrue((env.history_fx[-1] - env.history_fx[-2]) == state[1])

    def test_step(self):
        for config in onell_configs:
            env = self.make_env(config)
            state = env.reset()
            action = env.action_space.sample()
            self.assertTrue(action.shape[0] == len(env.action_var_names))
            state, reward, done, _ = env.step(action)
            self.assertTrue(issubclass(type(state), np.ndarray))
            self.assertTrue(len(env.state_var_names) == len(state))


# TestOneLLEnv().test_setup()
# TestOneLLEnv().test_reset()
# TestOneLLEnv().test_get_state()
# TestOneLLEnv().test_step()
