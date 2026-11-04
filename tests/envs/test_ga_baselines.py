"""Tests for the GA baseline policies and the baseline example script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from dacbench.benchmarks import GABenchmark
from dacbench.envs.env_utils.ga_baselines import (
    ALGORITHMS,
    _map_choice_to_index,
    random_policy,
    success_rule,
    theory_policy,
)
from dacbench.envs.theory import LeadingOne, OneMax

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTANCE_DIR = REPO_ROOT / "dacbench" / "instance_sets" / "ga"

PORTFOLIO = [1, 2, 4, 8, 16]


class FakeEnv:
    """Minimal environment interface used by the policies."""

    def __init__(
        self,
        fitness=0,
        n=50,
        algorithm="one_plus_one",
        action_choices=(1, 2, 4, 8, 16),
        problem=OneMax,
        c_step=0,
    ):
        """Initialise."""
        self.n = n
        self.algorithm = algorithm
        self.action_choices = (
            None if action_choices is None else list(action_choices)
        )
        self.problem = problem
        self.c_step = c_step
        self.x = SimpleNamespace(fitness=fitness)


def make_env(**config):
    """Create a GA environment with config overrides."""
    return GABenchmark(config=config).get_environment(test_env=True)


def load_example(module_name="run_ga_baselines_example"):
    """Import ``examples/run_ga_baselines.py`` as a module."""
    path = REPO_ROOT / "examples" / "run_ga_baselines.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    return example


class TestMapChoiceToIndex:
    """Tests for the portfolio mapping shared by the policies."""

    def test_listed_value_maps_to_its_index(self):
        """A value of the portfolio is returned as its exact index."""
        env = FakeEnv()
        for index, value in enumerate(PORTFOLIO):
            assert _map_choice_to_index(env, value) == index

    def test_nearest_value_in_log_space(self):
        """A value outside the portfolio maps to the nearest one in log space."""
        env = FakeEnv()
        assert _map_choice_to_index(env, 3) == 2  # |ln 3 - ln 4| < |ln 3 - ln 2|
        assert _map_choice_to_index(env, 6) == 3  # 8 is nearer than 4
        assert _map_choice_to_index(env, 1.4) == 0

    def test_clipping_to_the_portfolio(self):
        """Values beyond the portfolio are clipped to its ends."""
        env = FakeEnv(action_choices=[1, 2, 4])
        assert _map_choice_to_index(env, 64) == len(env.action_choices) - 1
        assert _map_choice_to_index(env, 0.1) == 0

    def test_continuous_action_space_rounds_the_value(self):
        """Without a portfolio the raw value is rounded to an integer."""
        env = FakeEnv(action_choices=None)
        assert _map_choice_to_index(env, 2.7) == 3
        assert _map_choice_to_index(env, 1.2) == 1


def drive(policy, env, improvements):
    """Feed ``improvements`` successful steps; return the last action index."""
    env.x.fitness, env.c_step = 0, 0
    action = policy(env, None)
    for step in range(1, improvements + 1):
        env.x.fitness, env.c_step = step, step
        action = policy(env, None)
    return action


class TestSuccessRule:
    """Tests for the one-fifth success rule."""

    def test_one_plus_one_increases_on_improvement(self):
        """R is multiplied by the factor after an improvement."""
        policy = success_rule("one_plus_one", action_choices=PORTFOLIO, initial_value=8)
        env = FakeEnv(fitness=0, c_step=0)
        assert policy(env, None) == 3  # 8
        env.x.fitness, env.c_step = 1, 1
        assert policy(env, None) == 4  # 12 -> nearest 16
        # already at the maximum: stays there on further improvements
        env.x.fitness, env.c_step = 2, 2
        assert policy(env, None) == 4

    def test_one_plus_one_decreases_on_non_improvement(self):
        """R is multiplied by factor**(-1/4) after a non-improvement."""
        policy = success_rule("one_plus_one", action_choices=PORTFOLIO, initial_value=8)
        env = FakeEnv(fitness=0, c_step=0)
        assert policy(env, None) == 3
        for step in range(1, 6):  # 5 failures: 8 * 1.5**(-5/4) ~ 4.8
            env.c_step = step
            action = policy(env, None)
        assert action == 2  # 4
        # already at the minimum: stays there on further non-improvements
        policy = success_rule("one_plus_one", action_choices=PORTFOLIO, initial_value=1)
        env = FakeEnv(fitness=0, c_step=1)
        assert policy(env, None) == 0
        for step in range(2, 8):
            env.c_step = step
            assert policy(env, None) == 0

    def test_lambda_rules_decrease_on_improvement(self):
        """For the lambda algorithms an improvement reduces lambda."""
        for algorithm in ["one_plus_lambda", "one_plus_lambda_lambda"]:
            policy = success_rule(algorithm, action_choices=PORTFOLIO, initial_value=8)
            env = FakeEnv(fitness=0, algorithm=algorithm, c_step=0)
            assert policy(env, None) == 3
            env.x.fitness, env.c_step = 1, 1
            assert policy(env, None) == 2  # 8 / 1.5 = 5.33 -> nearest 4
            # non-improvement: 5.33 * 1.5**0.25 = 5.90 -> nearest 8
            env.c_step = 2
            assert policy(env, None) == 3

    def test_lambda_rules_increase_on_non_improvement(self):
        """A run of non-improvements grows lambda for the lambda algorithms."""
        policy = success_rule(
            "one_plus_lambda_lambda", action_choices=PORTFOLIO, initial_value=1
        )
        env = FakeEnv(fitness=0, algorithm="one_plus_lambda_lambda", c_step=0)
        assert policy(env, None) == 0
        actions = []
        for step in range(1, 31):
            env.c_step = step
            actions.append(policy(env, None))
        assert actions[-1] == 4  # reached the maximum of the portfolio
        assert actions == sorted(actions)  # monotone growth without successes

    def test_stays_in_range_all_algorithms(self):
        """The returned index is always a valid action index."""
        rng = np.random.default_rng(0)
        for algorithm in ALGORITHMS:
            policy = success_rule(algorithm, action_choices=PORTFOLIO)
            env = FakeEnv(fitness=0, algorithm=algorithm, c_step=0)
            policy(env, None)
            fitness = 0
            for step in range(1, 100):
                fitness += int(rng.integers(2))  # improve or not
                env.x.fitness, env.c_step = fitness, step
                action = policy(env, None)
                assert 0 <= action < len(PORTFOLIO)

    def test_reset_restores_the_initial_value(self):
        """``reset()`` restores the initial value at a nonzero step count."""
        policy = success_rule("one_plus_one", action_choices=PORTFOLIO, initial_value=2)
        env = FakeEnv(fitness=0, c_step=0)
        # four improvements: 2 * 1.5**4 = 10.1 -> portfolio 8
        assert drive(policy, env, 4) == 3
        policy.reset()
        # c_step is nonzero and the fitness does not drop, so only the explicit
        # reset can bring the value back down from 10.1
        env.x.fitness, env.c_step = 4, 9
        assert policy(env, None) == 1

    def test_reset_on_first_step(self):
        """A call at ``c_step == 0`` restarts, without an explicit reset."""
        policy = success_rule("one_plus_one", action_choices=PORTFOLIO, initial_value=2)
        env = FakeEnv(fitness=0, c_step=0)
        assert drive(policy, env, 4) == 3  # value 10.1
        # same fitness as before (no drop) but a new episode: without the c_step
        # check the value would fall to 9.1 and stay at portfolio 8
        env.x.fitness, env.c_step = 4, 0
        assert policy(env, None) == 1

    def test_reset_when_the_fitness_drops(self):
        """A fitness drop restarts even if ``c_step`` looks like a new step."""
        policy = success_rule("one_plus_one", action_choices=PORTFOLIO, initial_value=2)
        env = FakeEnv(fitness=0, c_step=0)
        assert drive(policy, env, 4) == 3  # value 10.1
        # a drop cannot happen within an episode; without this check the failure
        # update would keep the value at 9.1, i.e. portfolio 8
        env.x.fitness, env.c_step = 0, 9
        assert policy(env, None) == 1

    def test_value_is_clipped_after_saturation(self):
        """The tracked value is clipped, so it recovers faster than a free one."""
        policy = success_rule(
            "one_plus_one", action_choices=[1, 2, 8], initial_value=8, factor=1.5
        )
        env = FakeEnv(fitness=0, c_step=0, action_choices=[1, 2, 8])
        assert policy(env, None) == 2  # 8, the maximum
        for step in range(1, 7):  # six improvements: 8 * 1.5**6 = 91 unclipped
            env.x.fitness, env.c_step = step, step
            assert policy(env, None) == 2
        # eight failures: the clipped value is 8 / 1.5**2 = 3.56 -> 2,
        # an unclipped value would still be 40.5 -> 8
        for step in range(7, 15):
            env.x.fitness, env.c_step = 6, step
            action = policy(env, None)
        assert action == 1

    def test_continuous_action_space(self):
        """Without a portfolio the raw mutation strength is returned."""
        policy = success_rule("one_plus_one", initial_value=1)
        env = FakeEnv(fitness=0, c_step=0, n=20, action_choices=None)
        assert policy(env, None) == 1
        actions = []
        for step in range(1, 12):  # 1 * 1.5**11 > 20, so the clip is reached
            env.x.fitness, env.c_step = step, step
            actions.append(policy(env, None))
        assert all(1 <= action <= 20 for action in actions)
        assert actions[-1] == 20
        assert actions == sorted(actions)

    def test_invalid_arguments(self):
        """Unknown algorithms, weak factors and empty portfolios are rejected."""
        with pytest.raises(ValueError, match="unknown algorithm"):
            success_rule("one_plus_two")
        with pytest.raises(ValueError, match="factor"):
            success_rule("one_plus_one", factor=1.0)
        with pytest.raises(ValueError, match="must not be empty"):
            success_rule("one_plus_one", action_choices=[])

    def test_runs_in_real_env_for_all_algorithms(self):
        """End-to-end: one short real episode per algorithm."""
        for algorithm in ALGORITHMS:
            env = make_env(
                algorithm=algorithm,
                action_choices=[1, 2, 4],
                instance_set_path=str(INSTANCE_DIR / "onemax_50_random.csv"),
                cutoff=50,
            )
            policy = success_rule(algorithm, initial_value=2)
            state, _ = env.reset(seed=0)
            steps = 0
            while True:
                action = policy(env, state)
                assert 0 <= action < 3
                state, _, terminated, truncated, _ = env.step(action)
                steps += 1
                if terminated or truncated:
                    break
            assert steps > 0


class TestRandomPolicy:
    """Tests for the uniform random baseline."""

    def test_in_range_and_deterministic(self):
        """Indices are valid and reproducible for a fixed seed."""
        env = FakeEnv()
        first = random_policy(seed=0)
        second = random_policy(seed=0)
        seq_a = [first(env, None) for _ in range(50)]
        seq_b = [second(env, None) for _ in range(50)]
        assert seq_a == seq_b
        assert all(0 <= action < len(PORTFOLIO) for action in seq_a)
        assert len(set(seq_a)) > 1

    def test_reset_reseeds(self):
        """reset() restarts the same sequence."""
        policy = random_policy(seed=7)
        env = FakeEnv()
        seq = [policy(env, None) for _ in range(20)]
        policy.reset()
        assert [policy(env, None) for _ in range(20)] == seq

    def test_different_seeds_differ(self):
        """Different seeds give different sequences."""
        env = FakeEnv()
        a = [random_policy(seed=0)(env, None) for _ in range(50)]
        b = [random_policy(seed=1)(env, None) for _ in range(50)]
        assert a != b

    def test_continuous_action_space(self):
        """Without a portfolio a mutation strength in [1, n] is sampled."""
        env = FakeEnv(n=10, action_choices=None)
        policy = random_policy(seed=0)
        actions = [policy(env, None) for _ in range(50)]
        assert all(1 <= action <= 10 for action in actions)
        assert len(set(actions)) > 1


class TestTheoryPolicy:
    """The theory-derived baseline policies."""

    def test_ga_onemax_lambda_formula(self):
        """The fitness-dependent lambda is rounded and mapped to the portfolio."""
        policy = theory_policy("OneMax", "one_plus_lambda_lambda")
        env = FakeEnv(n=50, algorithm="one_plus_lambda_lambda")
        env.x.fitness = 0  # sqrt(50 / 50) = 1 -> lambda = 1
        assert policy(env, None) == 0
        env.x.fitness = 38  # sqrt(50 / 12) = 2.04 -> lambda = 2
        assert policy(env, None) == 1
        env.x.fitness = 49  # sqrt(50 / 1) = 7.07 -> nearest portfolio 8
        assert policy(env, None) == 3

    def test_ga_onemax_lambda_is_rounded_not_truncated(self):
        """The formula is rounded: sqrt(100/12) = 2.89 -> 3 -> portfolio 4."""
        env = FakeEnv(n=100, algorithm="one_plus_lambda_lambda")
        env.x.fitness = 88
        policy = theory_policy("OneMax", "one_plus_lambda_lambda")
        assert policy(env, None) == 2  # truncation would give 2 and index 1

    def test_ga_onemax_at_the_optimum_does_not_divide_by_zero(self):
        """f(x) = n is guarded by max(1, n - f(x))."""
        env = FakeEnv(n=50, algorithm="one_plus_lambda_lambda")
        env.x.fitness = 50
        policy = theory_policy("OneMax", "one_plus_lambda_lambda")
        assert policy(env, None) == 3  # sqrt(50) = 7.07 -> 8

    def test_ga_onemax_clips_to_the_largest_action(self):
        """A formula value above the portfolio is clipped to the largest action."""
        policy = theory_policy("OneMax", "one_plus_lambda_lambda")
        env = FakeEnv(
            n=200,
            algorithm="one_plus_lambda_lambda",
            action_choices=[1, 2, 4],
        )
        env.x.fitness = 199  # sqrt(200) = 14.14 -> clipped to 4
        assert policy(env, None) == len(env.action_choices) - 1

    def test_onemax_constants(self):
        """OneMax (1+1) and (1+lambda) use the published constant actions."""
        for algorithm in ["one_plus_one", "one_plus_lambda"]:
            policy = theory_policy("OneMax", algorithm)
            env = FakeEnv(n=100, algorithm=algorithm)
            env.x.fitness = 0
            assert policy(env, None) == 0  # r = 1 / lambda = 1
            env.x.fitness = 99  # the constant does not depend on f(x)
            assert policy(env, None) == 0
            assert policy.reset() is None

    def test_leadingone_one_plus_one_is_one(self):
        """LeadingOne (1+1) uses r = 1, not the nominal nearest value r = 2.

        The published fixed rate is p ~ 1.59/n, i.e. a nominal r = 1.59, but the
        environment conditions on flipping at least one bit, which raises the
        expected number of flipped bits to r / (1 - exp(-r)): 1.58 at r = 1 and
        2.31 at r = 2. r = 1 is therefore the rate the published analysis is
        about.
        """
        policy = theory_policy("LeadingOne", "one_plus_one")
        env = FakeEnv(problem=LeadingOne, n=100, algorithm="one_plus_one")
        assert policy(env, None) == 0

    def test_leadingone_one_plus_lambda_is_one(self):
        """LeadingOne (1+lambda) uses the smallest admissible population."""
        policy = theory_policy("LeadingOne", "one_plus_lambda")
        env = FakeEnv(problem=LeadingOne, n=100, algorithm="one_plus_lambda")
        assert policy(env, None) == 0

    def test_continuous_action_space_returns_the_strength(self):
        """Without a portfolio the raw mutation strength is returned."""
        policy = theory_policy("OneMax", "one_plus_one")
        env = FakeEnv(n=50, algorithm="one_plus_one", action_choices=None)
        assert policy(env, None) == 1

    def test_ga_on_leadingone_raises(self):
        """No published theory for the GA on LeadingOne."""
        with pytest.raises(ValueError, match="LeadingOne") as excinfo:
            theory_policy("LeadingOne", "one_plus_lambda_lambda")
        assert "success_rule" in str(excinfo.value)
        assert "measure the best fixed action" in str(excinfo.value)

    def test_unknown_combination_raises(self):
        """Unknown problem and algorithm are rejected."""
        with pytest.raises(ValueError, match="unknown problem"):
            theory_policy("Trap", "one_plus_one")
        with pytest.raises(ValueError, match="unknown algorithm"):
            theory_policy("OneMax", "two_plus_two")

    @pytest.mark.parametrize(
        ("problem", "algorithm"),
        [
            ("OneMax", "one_plus_one"),
            ("OneMax", "one_plus_lambda"),
            ("OneMax", "one_plus_lambda_lambda"),
            ("LeadingOne", "one_plus_one"),
            ("LeadingOne", "one_plus_lambda"),
        ],
    )
    def test_action_is_valid_in_a_real_env(self, problem, algorithm):
        """Every returned index is a valid action index of a real environment."""
        env = make_env(
            problem=problem,
            algorithm=algorithm,
            action_choices=PORTFOLIO,
            instance_set_path=str(
                INSTANCE_DIR
                / f"{'onemax' if problem == 'OneMax' else 'leadingones'}_50.csv"
            ),
            cutoff=50,
        )
        env.reset(seed=0)
        policy = theory_policy(problem, algorithm)
        for fitness in (0, 25, 49):
            env.x.fitness = fitness
            action = policy(env, None)
            assert 0 <= action < len(env.action_choices)


class TestExampleScript:
    """Tests for examples/run_ga_baselines.py."""

    def test_policies_run_one_episode_in_the_example(self):
        """The example's episode loop finishes with every policy."""
        example = load_example()
        seeds = [0]
        for name in example.POLICIES:
            evals, solved = example.run_policy(
                "OneMax",
                20,
                "one_plus_one",
                lambda seed, nm=name: example.make_policy(nm, "OneMax", "one_plus_one", seed),
                seeds,
            )
            assert all(value >= 1 for value in evals)
            assert all(solved)

    def test_best_fixed_is_the_reference(self):
        """``best_fixed`` has ratio 1.00 and the others are relative to it."""
        example = load_example()
        rows, unavailable = example.evaluate(
            ["OneMax"], [20], ["one_plus_one"], [0, 1]
        )
        assert unavailable == []
        best = rows[0]
        assert best["policy"].startswith("best_fixed(r=")
        assert best["ratio"] == pytest.approx(1.0)
        by_policy = {row["policy"]: row for row in rows}
        assert set(by_policy) == {
            best["policy"],
            "theory",
            "success_rule",
            "random",
        }
        for row in rows[1:]:
            assert row["ratio"] == pytest.approx(row["mean_evals"] / best["mean_evals"])

    def test_missing_theory_policy_still_has_ratios(self):
        """A combination without a theory policy is still compared to a reference."""
        example = load_example()
        rows, unavailable = example.evaluate(
            ["LeadingOne"], [50], ["one_plus_lambda_lambda"], [0]
        )
        assert len(unavailable) == 1
        assert rows[0]["policy"].startswith("best_fixed(r=")
        assert {row["policy"] for row in rows[1:]} == {"success_rule", "random"}
        assert all(row["ratio"] is not None for row in rows)

    def test_all_policies_see_the_same_seeds_and_instances(self):
        """The example is paired: every policy meets the same (seed, instance) pairs."""
        example = load_example()
        seen = {}
        real_make_env = example.make_env

        def recording_make_env(problem, n, algorithm, action_choices=None):
            env = real_make_env(
                problem,
                n,
                algorithm,
                example.ACTION_CHOICES if action_choices is None else action_choices,
            )
            real_reset = env.reset

            def reset(seed=None, **kwargs):
                out = real_reset(seed=seed, **kwargs)
                seen.setdefault(seed, []).append(
                    (env.inst_id, int(env.x.fitness), env.x.data.tobytes())
                )
                return out

            env.reset = reset
            return env

        example.make_env = recording_make_env
        seeds = [0, 1]
        example.evaluate(["OneMax"], [20], ["one_plus_one"], seeds)
        # five constant actions plus the three adaptive policies, per seed
        for seed in seeds:
            observations = seen[seed]
            assert len(observations) == len(example.ACTION_CHOICES) + len(
                example.POLICIES
            )
            assert len(set(observations)) == 1

    def test_summary_names_only_the_selected_problems(self):
        """The footer lists the evaluated instance sets, not all of them."""
        example = load_example()
        lines = example.summary(["OneMax"], [0], [])
        text = "\n".join(lines)
        assert "onemax" in text
        assert "leadingones" not in text
        assert "single episode" in text

    def test_summary_reports_an_absent_theory_policy(self):
        """The footer names the configuration without a theory policy."""
        example = load_example()
        reason = "Error: no theory policy"
        lines = example.summary(
            ["LeadingOne"], [0], [("LeadingOne", 50, "one_plus_lambda_lambda", reason)]
        )
        text = "\n".join(lines)
        assert "LeadingOne n=50 one_plus_lambda_lambda" in text
        assert reason in text
