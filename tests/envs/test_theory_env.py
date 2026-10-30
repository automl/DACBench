from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np
import pytest
from dacbench.benchmarks import TheoryBenchmark
from dacbench.envs.theory import LeadingOne


def make_discrete_env(**config):
    """Build a discrete TheoryEnv on lo_rls_50 with config overrides."""
    bench = TheoryBenchmark(config={"instance_set_path": "lo_rls_50.csv", **config})
    return bench.get_environment()


class TestTheoryEnv(unittest.TestCase):
    def test_discrete_env(self):
        bench = TheoryBenchmark(
            config={
                "discrete_action": True,
                "action_choices": [1, 2, 4, 8],
                "instance_set_path": "lo_rls_50.csv",
            }
        )
        env = bench.get_environment()

        # check observation space
        s, _ = env.reset()  # default observation space: n, f(x)
        assert len(s) == 2
        assert s[0] == env.n
        assert s[1] == env.x.fitness

        # check action space
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 4

        # check instance-specific cutoff time
        assert env.max_evals == int(0.8 * env.n * env.n)

        # check initial solution for various instances
        for _i in range(len(env.instance_set)):
            if env.instance.initObj != "random":
                assert int(env.x.fitness) == int(env.instance.initObj)
            env.reset()

    def test_non_discrete_env(self):
        bench = TheoryBenchmark(
            config={
                "discrete_action": False,
                "min_action": 1,
                "max_action": 49,
                "instance_set_path": "lo_rls_50.csv",
            }
        )
        env = bench.get_environment()

        # check observation space
        s, _ = env.reset()  # default observation space: n, f(x)
        assert len(s) == 2
        assert s[0] == env.n
        assert s[1] == env.x.fitness

        # check action space
        assert isinstance(env.action_space, gym.spaces.Box)

        # check instance-specific cutoff time
        assert env.max_evals == int(0.8 * env.n * env.n)

        # check initial solution for various instances
        for _i in range(len(env.instance_set)):
            if env.instance.initObj != "random":
                assert int(env.x.fitness) == int(env.instance.initObj)
            env.reset()

        # check behaviour with out-of-range action
        s, r, terminated, truncated, info = env.step(
            100
        )  # a large negative reward will be returned and the epsidoe will end
        assert r < -1e4
        assert terminated or truncated

    def test_cutoff_truncates(self):
        """Reaching the step cutoff truncates the episode."""
        bench = TheoryBenchmark()
        bench.config.cutoff = 3
        env = bench.get_environment()
        env.reset(seed=0)
        # Start from the all-zeros string so three steps cannot reach the optimum.
        env.x.data[:] = False
        env.x.eval()

        for _ in range(3):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())

        assert not env.x.is_optimal()
        assert truncated
        assert not terminated

    def test_optimum_terminates(self):
        """Reaching the optimum terminates the episode."""
        bench = TheoryBenchmark()
        env = bench.get_environment()
        env.reset(seed=0)
        # LeadingOnes optimum: every bit set.
        env.x.data[:] = True
        env.x.eval()

        _, _, terminated, truncated, _ = env.step(env.action_space.sample())

        assert env.x.is_optimal()
        assert terminated
        assert not truncated
    def test_step_return_order(self):
        """step() returns (obs, reward, terminated, truncated, info)."""
        env = make_discrete_env()
        env.reset()
        env.max_evals = 1  # every step exhausts the evaluation budget
        _, _, terminated, truncated, _ = env.step(0)
        assert terminated is True
        assert truncated is False  # cutoff (1e6 steps) not reached

    def test_reward_logged_once_per_step(self):
        """Each step logs its reward exactly once, so R is the episode return."""
        env = make_discrete_env()
        env.reset()
        rewards = []
        terminated = truncated = False
        while not (terminated or truncated):
            _, r, terminated, truncated, info = env.step(0)
            rewards.append(r)
        assert len(env.log_reward) == env.c_step == len(rewards)
        assert info["values"]["R"] == pytest.approx(sum(rewards))

    def test_reward_logged_once_out_of_range(self):
        """An out-of-range action in training logs its reward exactly once."""
        bench = TheoryBenchmark(
            config={
                "discrete_action": False,
                "min_action": 1,
                "max_action": 49,
                "instance_set_path": "lo_rls_50.csv",
            }
        )
        env = bench.get_environment()
        env.reset()
        _, r, terminated, _, info = env.step(100)
        assert terminated
        assert env.log_reward == [r]
        assert info["values"]["R"] == r

    def test_imp_div_evals_new_reward(self):
        """'imp_div_evals_new' is (improvement - 0.5) / evals."""
        env = make_discrete_env(reward_choice="imp_div_evals_new")
        env.reset()
        f_before = env.x.fitness
        _, r, _, _, _ = env.step(0)  # flip 1 bit -> 1 evaluation
        assert r == pytest.approx((env.x.fitness - f_before - 0.5) / 1)

    def test_registered_env_maps_action_index(self):
        """gym.make('Theory-v0') maps an action index to action_choices."""
        env = gym.make("Theory-v0").unwrapped
        choices = env.config["action_choices"]
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == len(choices)
        env.reset(seed=0)
        env.step(2)
        assert env.history_r[-1] == choices[2]

    def test_continuous_warns_on_explicit_action_choices(self):
        """Explicit action_choices with continuous actions triggers a warning."""
        config = {
            "discrete_action": False,
            "min_action": 1,
            "max_action": 49,
            "instance_set_path": "lo_rls_50.csv",
        }
        with pytest.warns(UserWarning, match="action_choices"):
            TheoryBenchmark(config={**config, "action_choices": [1, 2]})


@pytest.mark.parametrize("init_obj", [0, 3, 5])
def test_leading_one_init_obj(init_obj):
    """LeadingOne with initObj has fitness initObj (initObj == n is optimal)."""
    x = LeadingOne(5, rng=np.random.default_rng(0), initObj=init_obj)
    assert x.fitness == init_obj
    assert x.is_optimal() == (init_obj == 5)


@pytest.mark.parametrize("init_obj", [-1, 6])
def test_leading_one_init_obj_out_of_range(init_obj):
    """LeadingOne rejects initObj outside [0, n]."""
    with pytest.raises(ValueError, match="initObj"):
        LeadingOne(5, rng=np.random.default_rng(0), initObj=init_obj)


if __name__ == "__main__":
    TestTheoryEnv().test_discrete_env()
    TestTheoryEnv().test_non_discrete_env()
