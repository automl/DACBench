"""Tests for the GA environment and the OneMax / BinaryProblem helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
from dacbench.benchmarks import GABenchmark
from dacbench.envs.theory import OneMax

GA_INSTANCE_DIR = (
    Path(__file__).resolve().parents[2] / "dacbench" / "instance_sets" / "ga"
)


def make_env(test_env=False, **config):
    """Create a GA env from config overrides."""
    return GABenchmark(config=config).get_environment(test_env=test_env)


def reset_to(env, instance_id, seed=0):
    """Reset env to a specific instance."""
    return env.reset(seed=seed, options={"instance_id": instance_id})


def evals_to_optimum(algorithm, seeds, n_instance=3):
    """Mean number of evaluations to the optimum on onemax_20.csv (initObj n//2)."""
    env = make_env(
        test_env=True,
        algorithm=algorithm,
        action_choices=[1],
        instance_set_path="onemax_20.csv",
    )
    evals = []
    for seed in seeds:
        reset_to(env, n_instance, seed=seed)
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(0)
            done = terminated or truncated
        assert env.x.is_optimal()
        evals.append(env.total_evals)
    return float(np.mean(evals))


class RecordingRng:
    """Wrap a numpy Generator; record choice() outputs and binomial() calls."""

    def __init__(self, seed):
        """Init."""
        self.rng = np.random.default_rng(seed)
        self.choices = []
        self.binomials = []  # (n, p, size, output)

    def binomial(self, n, p, size=None):
        """Delegate binomial and record the call."""
        out = self.rng.binomial(n, p, size=size)
        self.binomials.append((n, p, size, out))
        return out

    def random(self, *args, **kwargs):
        """Delegate random."""
        return self.rng.random(*args, **kwargs)

    def choice(self, *args, **kwargs):
        """Delegate choice and record the result."""
        out = self.rng.choice(*args, **kwargs)
        self.choices.append(np.sort(np.asarray(out)))
        return out


class ConstantFitnessOneMax(OneMax):
    """OneMax where every mutant / crossover child has the same fitness."""

    def get_fitness_after_flipping(self, locs):
        """Return constant fitness."""
        return 1000

    def get_fitness_after_crossover(self, xprime, locs_x, locs_xprime):
        """Return constant fitness."""
        return 1000


class TestGAEnv(unittest.TestCase):
    """Tests for GAEnv."""

    def test_discrete_env_defaults(self):
        """Default observation, action space and training cutoff."""
        env = make_env(action_choices=[1, 2, 4, 8])
        s, _ = env.reset()
        assert len(s) == 2
        assert s[0] == env.n
        assert s[1] == env.x.fitness
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 4
        assert env.max_evals == int(0.8 * env.n * env.n)

    def test_initial_fitness_matches_init_obj(self):
        """Every non-random instance starts with fitness == initObj."""
        files = sorted(GA_INSTANCE_DIR.glob("*.csv"))
        assert len(files) == 17
        for path in files:
            problem = "LeadingOne" if path.name.startswith("leading") else "OneMax"
            env = make_env(problem=problem, instance_set_path=path.name)
            df = pd.read_csv(path, index_col=0)
            for inst_id, row in df.iterrows():
                reset_to(env, inst_id)
                assert env.n == row["size"]
                if str(row["initObj"]) != "random":
                    assert env.x.fitness == int(row["initObj"]), (path.name, inst_id)

    def test_lambda_one_equivalence_in_distribution(self):
        """lambda=1 for the lambda algorithms is the (1+1) EA_{>0} with r=1."""
        seeds = range(300)
        ref = evals_to_optimum("one_plus_one", seeds)
        for algorithm in ["one_plus_lambda", "one_plus_lambda_lambda"]:
            mean = evals_to_optimum(algorithm, [s + 10_000 for s in seeds])
            assert abs(mean - ref) / ref < 0.15, (algorithm, mean, ref)

    def test_ollga_params(self):
        """Standard coupling p = lambda/n, c = 1/lambda."""
        env = make_env(algorithm="one_plus_lambda_lambda")
        env.reset()
        assert env.get_ollga_params(1) == (1 / env.n, 1.0)
        assert env.get_ollga_params(8) == (8 / env.n, 1 / 8)

    def test_one_plus_lambda_independent_strengths(self):
        """(1+lambda) EA: every child draws its own l_i ~ Bin(n, 1/n)."""
        n = 50
        for lbd in [4, 8]:
            env = make_env(
                algorithm="one_plus_lambda",
                action_choices=[lbd],
                instance_set_path="onemax_50.csv",
            )
            for seed in range(20):
                reset_to(env, 3, seed=seed)
                rng = RecordingRng(seed)
                env.np_random = rng
                env.step(0)
                assert all(
                    (bn, p, size) == (n, 1 / n, None)
                    for bn, p, size, _ in rng.binomials
                )
                # one non-zero draw per child, zeros are resampled
                nonzero = [out for *_, out in rng.binomials if out > 0]
                assert len(nonzero) == lbd
                assert rng.binomials[-1][3] > 0
                # each child flips exactly its own number of bits
                assert [len(c) for c in rng.choices] == nonzero

    def test_ollga_operator_structure(self):
        """(1+(lambda,lambda)) GA: shared l ~ Bin(n, lambda/n), bias c = 1/lambda."""
        n = 50
        for lbd in [4, 8]:
            env = make_env(
                algorithm="one_plus_lambda_lambda",
                action_choices=[lbd],
                instance_set_path="onemax_50.csv",
            )
            for seed in range(20):
                reset_to(env, 3, seed=seed)
                rng = RecordingRng(seed)
                env.np_random = rng
                env.step(0)
                *mutation, crossover = rng.binomials
                # mutation phase: scalar draws with p = lambda/n until l > 0
                assert all(
                    (bn, p, size) == (n, lbd / n, None) for bn, p, size, _ in mutation
                )
                assert all(out == 0 for *_, out in mutation[:-1])
                length = mutation[-1][3]
                assert length > 0
                # all lambda mutants flip the same number l of bits
                assert [len(c) for c in rng.choices[:lbd]] == [length] * lbd
                # crossover phase: lambda draws with bias c = 1/lambda
                bn, p, size, ls = crossover
                assert (bn, p, size) == (n, 1 / lbd, lbd)
                assert [len(c) for c in rng.choices[lbd:]] == list(ls)

    def test_ollga_mean_evals_regression(self):
        """Pinned: lambda=3, OneMax n=50 from n/2 needs ~311 evaluations on average."""
        env = make_env(
            test_env=True,
            algorithm="one_plus_lambda_lambda",
            action_choices=[3],
            instance_set_path="onemax_50.csv",
        )
        evals = []
        for seed in range(250):
            reset_to(env, 3, seed=seed)  # initObj 25
            done = False
            while not done:
                _, _, terminated, truncated, _ = env.step(0)
                done = terminated or truncated
            assert env.x.is_optimal()
            evals.append(env.total_evals)
        # reference: independent implementation ~311; this seed set gives 309.752
        assert abs(np.mean(evals) - 311) / 311 < 0.1, np.mean(evals)

    def test_eval_counting(self):
        """One generation uses 1 / lambda / <= 2 lambda evaluations."""
        lbd = 4
        expected = {"one_plus_one": (1, 1), "one_plus_lambda": (lbd, lbd)}
        expected["one_plus_lambda_lambda"] = (lbd, 2 * lbd)
        for algorithm, (low, high) in expected.items():
            env = make_env(algorithm=algorithm, action_choices=[1, lbd])
            for seed in range(20):
                reset_to(env, 1, seed=seed)  # initObj 0
                before = env.total_evals
                env.step(1)
                assert low <= env.total_evals - before <= high, algorithm
        env = make_env(
            algorithm="one_plus_lambda_lambda",
            action_choices=[lbd],
            count_different_inds_only=False,
        )
        for seed in range(20):
            reset_to(env, 1, seed=seed)
            env.step(0)
            assert env.total_evals == 1 + 2 * lbd

    def test_crossover_repairs_mutant(self):
        """Crossover combines the improving bit of x' with the rest of x."""
        n, a, c = 20, 0, 5
        successes = 0
        for seed in range(100):
            rng = np.random.default_rng(seed)
            x = OneMax(n, rng=rng, initObj=n)
            x.data[[a, 1]] = False  # zeros at a and 1
            x.eval()
            xprime = x.flip(np.array([a, c]))  # one improving, one worsening bit
            assert xprime.fitness == x.fitness
            child, fitness, _ = x.crossover(xprime, 0.5, n_childs=20, rng=rng)
            assert fitness == child.fitness
            if fitness == x.fitness + 1:
                assert child.data[a]
                assert child.data[c]
                successes += 1
        assert successes >= 95

    def test_out_of_range_continuous(self):
        """Value r > n: training env terminates, test env clips."""
        cfg = {"discrete_action": False, "instance_set_path": "onemax_50.csv"}
        env = make_env(**cfg)
        reset_to(env, 1)
        before = env.total_evals
        _, r, terminated, _, _ = env.step(100)
        assert r <= -1e4
        assert terminated
        assert env.total_evals == before

        env = make_env(test_env=True, **cfg)
        reset_to(env, 3)  # initObj n//2: flipping all bits does not hit the optimum
        _, r, terminated, truncated, _ = env.step(100)
        assert r > -1e4
        assert not terminated
        assert not truncated
        assert env.history_r[-1] == 50
        assert env.total_evals == 2

    def test_non_finite_continuous(self):
        """NaN / inf: training env terminates, test env clips inf, rejects NaN."""
        cfg = {"discrete_action": False, "instance_set_path": "onemax_50.csv"}
        for action in [np.nan, np.inf, -np.inf]:
            env = make_env(**cfg)
            reset_to(env, 1)
            before = env.total_evals
            _, r, terminated, _, info = env.step(action)
            assert r <= -1e4
            assert terminated
            assert env.total_evals == before
            assert info["values"]["log_r"] == [0]

        env = make_env(test_env=True, **cfg)
        reset_to(env, 3)
        for action, expected in [(np.inf, 50), (-np.inf, 1)]:
            before = env.total_evals
            _, r, terminated, truncated, _ = env.step(action)
            assert r > -1e4
            assert not terminated
            assert not truncated
            assert env.history_r[-1] == expected
            assert env.total_evals == before + 1
        with pytest.raises(ValueError, match="NaN"):
            env.step(np.nan)

    def test_out_of_range_discrete(self):
        """Choice value > n or invalid index: terminate (train) / clip (test)."""
        cfg = {"action_choices": [1, 100], "instance_set_path": "onemax_50.csv"}
        for action in [1, 2, np.int64(1), [1], np.array([1])]:
            env = make_env(**cfg)
            reset_to(env, 1)
            _, r, terminated, _, _ = env.step(action)
            assert r <= -1e4
            assert terminated

        env = make_env(test_env=True, **cfg)
        reset_to(env, 3)  # initObj n//2: flipping all bits does not hit the optimum
        for action in [1, 2]:
            _, r, terminated, truncated, _ = env.step(action)
            assert r > -1e4
            assert not terminated
            assert not truncated
            assert env.history_r[-1] == 50

    def test_continuous_lambda_rejected(self):
        """A continuous action space is not allowed for lambda algorithms."""
        for algorithm in ["one_plus_lambda", "one_plus_lambda_lambda"]:
            with pytest.raises(ValueError, match="discrete"):
                GABenchmark(config={"algorithm": algorithm, "discrete_action": False})
            bench = GABenchmark(config={"algorithm": algorithm})
            bench.config.discrete_action = False
            with pytest.raises(ValueError, match="discrete"):
                bench.get_environment()

    def test_imp_div_evals_new(self):
        """Regression: 'imp_div_evals_new' is supported and equals imp_div_evals."""
        rewards = []
        for choice in ["imp_div_evals", "imp_div_evals_new"]:
            env = make_env(
                algorithm="one_plus_lambda",
                reward_choice=choice,
                action_choices=[4],
            )
            reset_to(env, 1)
            f_before = env.x.fitness
            _, r, _, _, _ = env.step(0)
            assert r == pytest.approx((env.x.fitness - f_before - 0.5) / 4)
            rewards.append(r)
        assert rewards[0] == pytest.approx(rewards[1])

    def test_invalid_algorithm_and_obs_name(self):
        """Unknown algorithm or observation variable names are rejected."""
        with pytest.raises(AssertionError, match="algorithm"):
            make_env(algorithm="one_plus_two")
        for name in ["foo", "r_{t-5}", "delta_f(x)_{t-1}"]:
            with pytest.raises(ValueError, match="invalid state variable"):
                make_env(observation_description=f"n, {name}")

    def test_observation_variables(self):
        """All observation variables are computed from the histories."""
        env = make_env(
            test_env=True,
            action_choices=[2, 4],
            observation_description=(
                "n, f(x), delta_f(x), f(x)/n, r, lambda, r_{t-1}, lambda_{t-0}, "
                "f(x)_{t-1}"
            ),
        )
        reset_to(env, 1)
        env.step(0)
        s, _, _, _, _ = env.step(1)
        fx = env.history_fx
        assert list(s) == [50, fx[-1], fx[-1] - fx[-2], fx[-1] / 50, 4, 4, 2, 4, fx[-2]]
        assert env.observation_space.shape == (9,)

    def test_return_order_and_logging(self):
        """(state, reward, terminated, truncated, info); one log entry per step."""
        env = make_env(test_env=True, instance_set_path="onemax_20_easy.csv")
        env.reset(seed=0)
        steps, terminated, truncated = 0, False, False
        while not (terminated or truncated):
            _, reward, terminated, truncated, info = env.step(0)
            steps += 1
        assert terminated
        assert not truncated
        assert env.x.is_optimal()
        assert len(env.log_reward) == steps
        assert len(info["values"]["log_reward"]) == steps
        assert info["values"]["log_reward"][-1] == reward

        # training env: evaluation budget 0.8*n^2, step cutoff 3
        env = make_env(cutoff=3, instance_set_path="onemax_200.csv")
        reset_to(env, 1)
        for _ in range(3):
            _, _, terminated, truncated, info = env.step(0)
        assert truncated
        assert not terminated
        assert info["values"]["steps"] == 3
        assert len(info["values"]["log_reward"]) == 3

    def test_reward_choices(self):
        """imp_minus_evals_scaling = (delta f - evals) / n."""
        env = make_env(
            algorithm="one_plus_lambda",
            reward_choice="imp_minus_evals_scaling",
            action_choices=[4],
        )
        reset_to(env, 1)
        f_before = env.x.fitness
        _, r, _, _, _ = env.step(0)
        assert r == pytest.approx((env.x.fitness - f_before - 4) / env.n)


class TestBinaryProblem(unittest.TestCase):
    """Tests for OneMax and the BinaryProblem fixes."""

    def test_initialise_with_fixed_number_of_bits(self):
        """Regression: unset bits were never found (`self.data is False`)."""
        x = OneMax(20, rng=np.random.default_rng(0), initObj=5)
        x.initialise_with_fixed_number_of_bits(10, rng=np.random.default_rng(1))
        assert x.data.sum() == 10
        assert x.fitness == 10

    def test_mutate_uniform_tie_breaking(self):
        """Regression: mutate always returned the first of tied children."""
        n_childs, n_trials = 4, 400
        picks = np.zeros(n_childs)
        for seed in range(n_trials):
            rng = RecordingRng(seed)
            x = ConstantFitnessOneMax(50, rng=np.random.default_rng(seed), initObj=0)
            child, _, _ = x.mutate(0.2, n_childs=n_childs, rng=rng)
            chosen = np.flatnonzero(child.data)
            idx = next(
                i for i, c in enumerate(rng.choices) if np.array_equal(c, chosen)
            )
            picks[idx] += 1
        assert np.all(np.abs(picks / n_trials - 1 / n_childs) < 0.08), picks

    def test_crossover_uniform_tie_breaking(self):
        """Regression: crossover always returned the first of tied children."""
        n_childs, n_trials = 4, 400
        picks = np.zeros(n_childs)
        for seed in range(n_trials):
            rng = RecordingRng(seed)
            x = ConstantFitnessOneMax(50, rng=np.random.default_rng(seed), initObj=0)
            xprime = OneMax(50, rng=np.random.default_rng(seed), initObj=50)
            child, _, _ = x.crossover(xprime, 0.5, n_childs=n_childs, rng=rng)
            chosen = np.flatnonzero(child.data)
            idx = next(
                i for i, c in enumerate(rng.choices) if np.array_equal(c, chosen)
            )
            picks[idx] += 1
        assert np.all(np.abs(picks / n_trials - 1 / n_childs) < 0.08), picks

    def test_crossover_keeps_xprime_on_ties(self):
        """The incumbent x' is only replaced by strictly better children."""
        rng = np.random.default_rng(0)
        x = OneMax(30, rng=rng, initObj=15)
        xprime = x.flip(np.array([0]))
        xprime.data = x.data.copy()
        xprime.eval()
        child, _, n_evals = x.crossover(xprime, 0.5, n_childs=10, rng=rng)
        assert child is xprime
        assert n_evals == 0

    def test_onemax_closed_forms(self):
        """Closed-form fitness after flipping / crossover equals re-evaluation."""
        rng = np.random.default_rng(0)
        for _ in range(200):
            n = int(rng.integers(1, 60))
            x = OneMax(n, rng=rng)
            locs = rng.choice(n, size=int(rng.integers(1, n + 1)), replace=False)
            assert x.get_fitness_after_flipping(locs) == x.flip(locs).fitness
            xprime = OneMax(n, rng=rng)
            locs_xprime = rng.choice(n, size=int(rng.integers(0, n + 1)), replace=False)
            locs_x = np.full(n, True)
            locs_x[locs_xprime] = False
            expected = x.combine(xprime, locs_xprime).fitness
            assert x.get_fitness_after_crossover(xprime, locs_x, locs_xprime) == (
                expected
            )
            assert x.fitness == int(x.data.sum())

    def test_onemax_init_obj_range(self):
        """OneMax supports initObj in [0, n]."""
        for k in [0, 7, 20]:
            x = OneMax(20, rng=np.random.default_rng(k), initObj=k)
            assert x.fitness == k
            assert x.is_optimal() == (k == 20)
            assert x.get_optimal() == 20
