"""Tests for the GA baseline policies and the baseline example script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from dacbench.benchmarks import GABenchmark
from dacbench.envs.env_utils import ga_baselines
from dacbench.envs.env_utils.ga_baselines import (
    ALGORITHMS,
    best_static,
    best_static_portfolio,
    load_policy_table,
    policy_file_name,
    random_policy,
    success_rule,
)
from dacbench.envs.theory import LeadingOne, OneMax

REPO_ROOT = Path(__file__).resolve().parents[2]

# {f_x: [(action, choice, mean_evals), ...]}
SYNTHETIC_TABLE = {
    0: [(0, 1, 100.0), (1, 2, 50.0), (2, 4, 50.0), (3, 8, np.inf), (4, 16, 75.0)],
    10: [(0, 1, 40.0), (1, 2, np.inf), (2, 4, 20.0), (3, 8, np.inf), (4, 16, np.inf)],
    20: [(0, 1, np.inf), (1, 2, np.inf), (2, 4, np.inf), (3, 8, 30.0), (4, 16, np.inf)],
    30: [
        (0, 1, np.inf),
        (1, 2, np.inf),
        (2, 4, np.inf),
        (3, 8, np.inf),
        (4, 16, np.inf),
    ],
}
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
        self.action_choices = list(action_choices)
        self.problem = problem
        self.c_step = c_step
        self.x = SimpleNamespace(fitness=fitness)


def _means(bucket):
    """Return the mean_evals values (finite and inf) of one bucket."""
    return [mean for _, _, mean in SYNTHETIC_TABLE[bucket]]


def write_table(path, table=SYNTHETIC_TABLE):
    """Write a CSV in the derived-table contract format and return its path."""
    lines = ["f_x,action,choice,mean_evals,n_seeds,ci_low,ci_high"]
    for f_x in sorted(table):
        for action, choice, mean in table[f_x]:
            lines.append(
                f"{f_x},{action},{choice},{mean},10,{mean},{mean}"
                if np.isfinite(mean)
                else f"{f_x},{action},{choice},inf,10,-inf,inf"
            )
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def table_dir(tmp_path):
    """Directory with one synthetic table for every (problem, n, algorithm)."""
    for problem in ["onemax", "leadingones"]:
        for n in [50, 100, 200]:
            for algorithm in ALGORITHMS:
                write_table(tmp_path / f"optimal_policy_{problem}_{n}_{algorithm}.csv")
    return tmp_path


def make_env(**config):
    """Create a GA environment with config overrides."""
    return GABenchmark(config=config).get_environment(test_env=True)


class TestBestStatic:
    """Tests for the simulation-derived best_static policy."""

    def test_file_name_contract(self):
        """File names follow the documented scheme."""
        assert policy_file_name("OneMax", 50, "one_plus_one") == (
            "optimal_policy_onemax_50_one_plus_one.csv"
        )
        assert policy_file_name("LeadingOne", 200, "one_plus_lambda") == (
            "optimal_policy_leadingones_200_one_plus_lambda.csv"
        )
        with pytest.raises(ValueError, match="unknown problem"):
            policy_file_name("Trap", 50, "one_plus_one")
        with pytest.raises(ValueError, match="unknown algorithm"):
            policy_file_name("OneMax", 50, "two_plus_two")

    def test_picks_minimum_finite_row(self, table_dir):
        """Each bucket resolves to the row with the smallest finite mean_evals."""
        policy = best_static(
            problem="OneMax", n=50, algorithm="one_plus_one", instance_dir=table_dir
        )
        env = FakeEnv()
        expected = {0: 1, 10: 2, 20: 3}
        for f_x, index in expected.items():
            env.x.fitness = f_x
            action = policy(env, None)
            assert action == index
            # this is exactly the minimum finite row of the bucket
            rows = [
                (mean, action_)
                for action_, _, mean in SYNTHETIC_TABLE[f_x]
                if np.isfinite(mean)
            ]
            best = min(rows)
            assert action == best[1]
            assert SYNTHETIC_TABLE[f_x][action][2] == best[0]

    def test_never_picks_inf(self, table_dir):
        """Rows with mean_evals == inf are never selected, even in gaps."""
        policy = best_static(
            problem="OneMax", n=50, algorithm="one_plus_lambda", instance_dir=table_dir
        )
        env = FakeEnv(algorithm="one_plus_lambda")
        for f_x in range(41):
            env.x.fitness = f_x
            action = policy(env, None)
            assert 0 <= action < len(PORTFOLIO)
            # documented rule: nearest bucket with at least one finite row,
            # ties resolved towards the smaller bucket
            buckets = sorted(SYNTHETIC_TABLE)
            usable = [b for b in buckets if np.any(np.isfinite(_means(b)))]
            bucket = min(usable, key=lambda b: (abs(b - f_x), b))
            mean = SYNTHETIC_TABLE[bucket][action][2]
            assert np.isfinite(mean)
            assert mean == np.min(_means(bucket))

    def test_bucket_gaps(self, table_dir):
        """Gaps and values beyond the table resolve to the nearest bucket."""
        policy = best_static(
            problem="OneMax", n=50, algorithm="one_plus_one", instance_dir=table_dir
        )
        env = FakeEnv()
        env.x.fitness = 5  # equidistant to 0 and 10 -> smaller bucket
        assert policy(env, None) == 1
        env.x.fitness = 6  # closer to 10
        assert policy(env, None) == 2
        env.x.fitness = 24  # closest to 20
        assert policy(env, None) == 3
        env.x.fitness = 50  # above the last bucket with finite values
        assert policy(env, None) == 3

    def test_inference_from_env(self, table_dir, monkeypatch):
        """problem/n/algorithm are inferred from the environment if omitted.

        The two problems get tables with a different best action, so the test
        really distinguishes them, and the loaded files are checked to be the
        ones the environment selects.
        """
        leadingone_table = dict(SYNTHETIC_TABLE)
        # bucket 10 with another best action, so the two problems differ
        leadingone_table[10] = [
            (0, 1, 40.0),
            (1, 2, 10.0),
            (2, 4, 20.0),
            (3, 8, np.inf),
            (4, 16, np.inf),
        ]
        write_table(
            table_dir / "optimal_policy_leadingones_100_one_plus_one.csv",
            leadingone_table,
        )

        loaded = []
        real_load = ga_baselines.load_policy_table

        def recording_load(path):
            loaded.append(Path(path).name)
            return real_load(path)

        monkeypatch.setattr(ga_baselines, "load_policy_table", recording_load)

        policy = best_static(instance_dir=table_dir)
        env = FakeEnv(fitness=10, n=100, algorithm="one_plus_one")
        assert policy(env, None) == 2  # OneMax: action 2 (mean 20)
        env.problem = LeadingOne  # other problem, other file name
        assert policy(env, None) == 1  # LeadingOne: action 1 (mean 10)
        assert loaded == [
            "optimal_policy_onemax_100_one_plus_one.csv",
            "optimal_policy_leadingones_100_one_plus_one.csv",
        ]

    def test_maps_choice_to_env_portfolio(self, table_dir):
        """A table choice outside the env portfolio maps to the nearest in log space."""
        policy = best_static(
            problem="OneMax", n=50, algorithm="one_plus_one", instance_dir=table_dir
        )
        env = FakeEnv(fitness=20, action_choices=[1, 2, 4, 8])  # table choice 8 -> 8
        assert policy(env, None) == 3
        env.action_choices = [1, 2, 4]  # table choice 8 -> nearest is 4
        assert policy(env, None) == 2

    def test_unknown_combination_raises(self, tmp_path):
        """A missing table raises a clear FileNotFoundError."""
        policy = best_static(
            problem="OneMax", n=50, algorithm="one_plus_one", instance_dir=tmp_path
        )
        with pytest.raises(FileNotFoundError, match="optimal_policy_onemax_50"):
            policy(FakeEnv(), None)

    def test_explicit_policy_path(self, tmp_path):
        """policy_path overrides the naming scheme."""
        path = write_table(tmp_path / "any_name.csv")
        policy = best_static(policy_path=path)
        assert policy(FakeEnv(fitness=10), None) == 2

    def test_missing_required_column(self, tmp_path):
        """A malformed table is rejected with a clear error."""
        path = tmp_path / "broken.csv"
        path.write_text("f_x,action\n0,0\n")
        with pytest.raises(ValueError, match="missing the column"):
            load_policy_table(path)
        with pytest.raises(FileNotFoundError):
            load_policy_table(tmp_path / "does_not_exist.csv")

    def test_portfolio_from_table(self, table_dir):
        """best_static_portfolio returns the sorted shipped portfolio."""
        assert (
            best_static_portfolio("OneMax", 50, "one_plus_one", instance_dir=table_dir)
            == PORTFOLIO
        )

    def test_real_table_if_present(self):
        """The shipped tables (if generated) are readable and usable."""
        table_dir = REPO_ROOT / "dacbench" / "instance_sets" / "ga"
        path = table_dir / "optimal_policy_onemax_50_one_plus_one.csv"
        if not path.is_file():
            pytest.skip("derived reference table not generated in this checkout")
        policy = best_static(
            problem="OneMax", n=50, algorithm="one_plus_one", instance_dir=table_dir
        )
        env = FakeEnv(fitness=0, n=50)
        assert 0 <= policy(env, None) < len(PORTFOLIO)


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

    def test_reset_between_episodes(self):
        """A new episode restores the initial value (explicit and automatic)."""
        policy = success_rule("one_plus_one", action_choices=PORTFOLIO, initial_value=4)
        env = FakeEnv(fitness=0, c_step=0)
        assert policy(env, None) == 2
        env.x.fitness, env.c_step = 1, 1
        assert policy(env, None) == 3
        # explicit reset
        policy.reset()
        env.x.fitness, env.c_step = 5, 0
        assert policy(env, None) == 2
        # automatic reset on c_step == 0
        env.x.fitness, env.c_step = 6, 1
        assert policy(env, None) == 3
        env.x.fitness, env.c_step = 0, 0
        assert policy(env, None) == 2
        # automatic reset when the fitness drops although c_step is stale
        env.x.fitness, env.c_step = 7, 2
        assert policy(env, None) == 3
        env.x.fitness = 1
        assert policy(env, None) == 2

    def test_invalid_arguments(self):
        """Unknown algorithms and update strengths are rejected."""
        with pytest.raises(ValueError, match="unknown algorithm"):
            success_rule("one_plus_two")
        with pytest.raises(ValueError, match="factor"):
            success_rule("one_plus_one", factor=1.0)

    def test_runs_in_real_env_for_all_algorithms(self):
        """End-to-end: one short real episode per algorithm."""
        for algorithm in ALGORITHMS:
            env = make_env(
                algorithm=algorithm,
                action_choices=[1, 2, 4],
                instance_set_path="onemax_50_random.csv",
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


class TestExampleScript:
    """Smoke test for examples/run_ga_baselines.py."""

    def test_core_loop_one_episode_per_policy(self, tmp_path):
        """The example's episode loop finishes with every policy."""
        path = REPO_ROOT / "examples" / "run_ga_baselines.py"
        spec = importlib.util.spec_from_file_location("run_ga_baselines_example", path)
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)

        # tiny table: action 0 (r = 1) is best everywhere, so every episode solves
        tiny = {0: [(0, 1, 10.0), (1, 2, 20.0)]}
        csv_path = write_table(tmp_path / "synthetic.csv", tiny)
        env = example.make_env("OneMax", 20, "one_plus_one", PORTFOLIO)
        policies = [
            success_rule("one_plus_one", action_choices=PORTFOLIO),
            random_policy(seed=0),
            best_static(policy_path=csv_path),
        ]
        for policy in policies:
            n_evals, solved = example.run_episode(env, policy, seed=0)
            assert n_evals >= 1
            assert solved
        # the reported table row format
        rows, missing = example.evaluate(
            ["OneMax"], [20], ["one_plus_one"], [0], ["success_rule", "random"]
        )
        assert missing
        assert {row["policy"] for row in rows} == {"success_rule", "random"}
        assert all(row["problem"] == "OneMax" and row["n"] == 20 for row in rows)

    def test_all_policies_see_the_same_seeds_and_instances(self):
        """The example is paired: every policy meets the same (seed, instance) pairs."""
        path = REPO_ROOT / "examples" / "run_ga_baselines.py"
        spec = importlib.util.spec_from_file_location("run_ga_baselines_paired", path)
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)

        seen = {}
        real_make_env = example.make_env

        def recording_make_env(problem, n, algorithm, action_choices):
            env = real_make_env(problem, n, algorithm, action_choices)
            real_reset = env.reset

            def reset(seed=None, **kwargs):
                out = real_reset(seed=seed, **kwargs)
                seen.setdefault(seed, []).append(env.inst_id)
                return out

            env.reset = reset
            return env

        example.make_env = recording_make_env
        seeds = [0, 1, 2]
        rows, _skipped = example.evaluate(
            ["OneMax"], [50], ["one_plus_one"], seeds, ["success_rule", "random"]
        )
        assert len(rows) == 2
        # each seed was reset once per policy, and the instances matched
        for seed in seeds:
            instances = seen[seed]
            assert len(instances) == 2
            assert len(set(instances)) == 1
