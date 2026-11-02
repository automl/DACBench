"""Tests for the GA baseline policies and the baseline example script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from dacbench.benchmarks import GABenchmark
from dacbench.envs.env_utils import ga_baselines
from dacbench.envs.env_utils.ga_baselines import (
    ALGORITHMS,
    EXACT_COMBINATIONS,
    REFERENCE_SIZES,
    best_static,
    best_static_portfolio,
    gap,
    load_policy_table,
    optimal_reference,
    policy_file_name,
    random_policy,
    success_rule,
    theory_policy,
)
from dacbench.envs.theory import LeadingOne, OneMax

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTANCE_DIR = REPO_ROOT / "dacbench" / "instance_sets" / "ga"

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


class TestOptimalReference:
    """The exact optimality-gap reference of the six exact combinations."""

    def test_exact_numbers_one_plus_one_n50(self):
        """V[0], the best fixed action and its exact cost of the (1+1) EA."""
        reference = optimal_reference("OneMax", 50, "one_plus_one", start_f=0)
        assert reference["available"] is True
        assert reference["exact"] is True
        assert reference["source"] == "exact DP"
        assert reference["start_f"] == 0
        assert reference["optimal_evals"] == pytest.approx(277.966676273948, rel=1e-9)
        assert reference["best_fixed_action"] == 0
        assert reference["best_fixed_choice"] == 1
        assert reference["best_fixed_evals"] == pytest.approx(
            308.798959959725, rel=1e-9
        )

    @pytest.mark.parametrize("n", REFERENCE_SIZES)
    def test_one_plus_lambda_ratio_is_exactly_one(self, n):
        """For the (1+lambda) EA the optimum is the constant action lambda = 1."""
        reference = optimal_reference("onemax", n, "one_plus_lambda", start_f=0)
        assert reference["best_fixed_choice"] == 1
        assert reference["optimal_evals"] / reference["best_fixed_evals"] == (
            pytest.approx(1.0, rel=1e-12)
        )

    def test_default_start_is_the_hardest_bucket(self):
        """start_f=None means the hardest start, f = 0."""
        default = optimal_reference("OneMax", 100, "one_plus_one")
        explicit = optimal_reference("OneMax", 100, "one_plus_one", start_f=0)
        assert default == explicit

    @pytest.mark.parametrize(("problem", "n", "algorithm"), EXACT_COMBINATIONS)
    def test_value_is_the_minimum_policy_column(self, problem, n, algorithm):
        """V[start_f] equals min_a V(start_f, a), the exact policy-table column.

        The value ``V`` is the minimum of "play ``a`` once, then optimally", not
        the minimum of "always play ``a``": the latter (``W``) is only an upper
        bound, with equality when the optimal policy is constant. The test also
        checks both sides against the shipped fixed-action CSV, so the API and
        the data cannot drift apart.
        """
        table = pd.read_csv(
            INSTANCE_DIR / f"optimal_policy_{problem}_{n}_{algorithm}.csv"
        )
        fixed = pd.read_csv(INSTANCE_DIR / "exact_fixed_action_counts.csv")
        block = fixed[(fixed["n"] == n) & (fixed["algorithm"] == algorithm)]
        for start_f in range(n + 1):
            reference = optimal_reference(problem, n, algorithm, start_f)
            column = table.loc[table["f_x"] == start_f, "mean_evals"].min()
            assert reference["optimal_evals"] == pytest.approx(float(column), rel=1e-12)
            rows = block[(block["start_f"] == start_f) & (block["action"] >= 0)]
            best_row = rows.loc[rows["expected_evals"].idxmin()]
            assert reference["best_fixed_action"] == int(best_row["action"])
            assert reference["best_fixed_choice"] == pytest.approx(
                float(best_row["choice"])
            )
            assert reference["best_fixed_evals"] == pytest.approx(
                float(best_row["expected_evals"]), rel=1e-12
            )
            # adapting after one generation is never worse than a constant
            assert reference["optimal_evals"] <= (reference["best_fixed_evals"] + 1e-9)

    def test_one_plus_one_value_is_below_best_fixed_at_the_hard_start(self):
        """The (1+1) EA really adapts at f=0; at f=n-1 the policy is constant.

        This documents the exact relation between the two references: ``V <
        min_a W_a`` where switching after the first generation helps (the lower
        buckets), and ``V = min_a W_a`` where the optimal policy is the constant
        ``r = 1`` (the upper buckets).
        """
        hard = optimal_reference("OneMax", 50, "one_plus_one", start_f=0)
        assert hard["optimal_evals"] < hard["best_fixed_evals"]
        upper = optimal_reference("OneMax", 50, "one_plus_one", start_f=49)
        assert upper["optimal_evals"] == pytest.approx(
            upper["best_fixed_evals"], rel=1e-12
        )

    @pytest.mark.parametrize(
        ("problem", "n", "algorithm", "needle"),
        [
            ("OneMax", 50, "one_plus_lambda_lambda", "arrangement"),
            ("OneMax", 50, "one_plus_lambda_lambda", "count_different_inds_only"),
            ("LeadingOne", 50, "one_plus_one", "suffix"),
            ("LeadingOne", 200, "one_plus_lambda", "suffix"),
        ],
    )
    def test_non_exact_combinations_have_no_number(self, problem, n, algorithm, needle):
        """Non-exact combinations return no number, only a true reason."""
        reference = optimal_reference(problem, n, algorithm, start_f=0)
        assert reference["available"] is False
        assert reference["exact"] is False
        assert reference["optimal_evals"] is None
        assert needle in reference["reason"]

    def test_non_exact_does_not_need_the_exact_table(self, tmp_path):
        """A non-exact combination is answered without reading the fixed table."""
        reference = optimal_reference(
            "OneMax", 50, "one_plus_lambda_lambda", instance_dir=tmp_path
        )
        assert reference["available"] is False

    @pytest.mark.parametrize("start_f", [-1, 51, 1.5, "0", True])
    def test_invalid_start_f_raises(self, start_f):
        """start_f must be an integer in [0, n]."""
        with pytest.raises(ValueError, match="start_f"):
            optimal_reference("OneMax", 50, "one_plus_one", start_f)
        with pytest.raises(ValueError, match="start_f"):
            gap(300, "OneMax", 50, "one_plus_one", start_f)

    def test_unknown_combination_raises_with_available_values(self):
        """Unknown problem, size or algorithm lists what is supported."""
        with pytest.raises(ValueError, match="unknown problem"):
            optimal_reference("Trap", 50, "one_plus_one")
        with pytest.raises(ValueError, match="unknown algorithm"):
            optimal_reference("OneMax", 50, "two_plus_two")
        with pytest.raises(ValueError, match=r"unknown n 75.*\(50, 100, 200\)"):
            optimal_reference("OneMax", 75, "one_plus_one")


class TestGap:
    """Arithmetic and censoring rules of the optimality gap."""

    def test_arithmetic(self):
        """The ratio is evals / optimal_evals, with the exact reference exposed."""
        reference = optimal_reference("OneMax", 50, "one_plus_one", start_f=0)
        result = gap(
            2 * reference["optimal_evals"],
            "OneMax",
            50,
            "one_plus_one",
            0,
        )
        assert result["evals"] == pytest.approx(2 * reference["optimal_evals"])
        assert result["reference_evals"] == reference["optimal_evals"]
        assert result["ratio"] == pytest.approx(2.0, rel=1e-12)
        assert result["exact"] is True
        assert result["solved"] is True
        assert result["reason"] is None

    def test_censored_run_has_no_gap(self):
        """An unsolved (cutoff) episode is censored: no ratio, a clear reason."""
        result = gap(10_000, "OneMax", 50, "one_plus_one", 0, solved=False)
        assert result["ratio"] is None
        assert result["reference_evals"] is not None
        assert result["exact"] is True
        assert result["solved"] is False
        assert "cutoff" in result["reason"]

    @pytest.mark.parametrize(
        ("problem", "n", "algorithm", "needle"),
        [
            ("OneMax", 50, "one_plus_lambda_lambda", "arrangement"),
            ("OneMax", 200, "one_plus_lambda_lambda", "crossover"),
            ("LeadingOne", 100, "one_plus_lambda_lambda", "suffix"),
            ("LeadingOne", 200, "one_plus_lambda", "suffix"),
        ],
    )
    def test_non_exact_has_no_ratio(self, problem, n, algorithm, needle):
        """No exact optimum: no ratio, no number, the reason is passed through."""
        result = gap(10_000, problem, n, algorithm, 0)
        assert result["ratio"] is None
        assert result["reference_evals"] is None
        assert result["exact"] is False
        assert result["solved"] is True
        assert needle in result["reason"]

    def test_optimal_start_has_no_gap(self):
        """start_f = n is already optimal: zero evaluations cannot be a ratio."""
        result = gap(0, "OneMax", 50, "one_plus_one", 50)
        assert result["ratio"] is None
        assert result["reference_evals"] == 0.0
        assert "already optimal" in result["reason"]


class TestTheoryPolicy:
    """The theory-derived reference policies."""

    def test_ga_onemax_lambda_formula(self):
        """The fitness-dependent lambda is mapped to the portfolio."""
        policy = theory_policy("OneMax", "one_plus_lambda_lambda")
        env = FakeEnv(n=50, algorithm="one_plus_lambda_lambda")
        env.x.fitness = 0  # sqrt(50 / 50) = 1 -> lambda = 1
        assert policy(env, None) == 0
        env.x.fitness = 38  # sqrt(50 / 12) = 2.04 -> lambda = 2
        assert policy(env, None) == 1
        env.x.fitness = 49  # sqrt(50 / 1) = 7.07 -> nearest portfolio 8
        assert policy(env, None) == 3

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
        """OneMax (1+1) and (1+lambda) use the exact-DP constant actions."""
        for algorithm in ["one_plus_one", "one_plus_lambda"]:
            policy = theory_policy("OneMax", algorithm)
            env = FakeEnv(n=100, algorithm=algorithm)
            env.x.fitness = 0
            assert policy(env, None) == 0  # r = 1 / lambda = 1
            env.x.fitness = 99  # the constant does not depend on f(x)
            assert policy(env, None) == 0
            assert policy.reset() is None

    def test_leadingone_constants(self):
        """LeadingOne uses the nearest portfolio value to p ~ 1.59/n."""
        one_plus_one = theory_policy("LeadingOne", "one_plus_one")
        env = FakeEnv(problem=LeadingOne, n=100, algorithm="one_plus_one")
        assert one_plus_one(env, None) == 1  # r = 2
        one_plus_lambda = theory_policy("LeadingOne", "one_plus_lambda")
        env.algorithm = "one_plus_lambda"
        assert one_plus_lambda(env, None) == 0  # lambda = 1

    def test_ga_on_leadingone_raises(self):
        """No published theory and no bucket model for the GA on LeadingOne."""
        with pytest.raises(ValueError, match="LeadingOne") as excinfo:
            theory_policy("LeadingOne", "one_plus_lambda_lambda")
        assert "suffix" in str(excinfo.value)
        assert "success_rule" in str(excinfo.value)

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
