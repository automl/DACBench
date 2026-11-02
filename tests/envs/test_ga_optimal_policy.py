"""Tests for the simulation-derived reference tables of the GA benchmark.

The tables live in ``dacbench/instance_sets/ga/optimal_policy_*.csv`` and are
produced by ``dacbench/instance_sets/ga/derive_optimal_policy.py`` (a standalone
script, see its module docstring).  The tests check the internal consistency of
the shipped tables, the scope of the derivation, the estimation internals, and
(cheaply) that the shipped policies have not drifted in full benchmark episodes.

Shipped are the 9 combinations whose derived policy was at most 1.05 times the
evaluations of the best fixed action on **both** paired seed sets A (seeds 0-11,
the selection protocol) and B (held-out seeds 1000-1011) and never significantly
worse: OneMax n=50 with the (1+1) EA and the (1+lambda) EA,
OneMax n=100 with the (1+1) EA and the (1+(lambda,lambda)) GA, all three OneMax
n=200 combinations, LeadingOne n=100 with the (1+1) EA and the
(1+(lambda,lambda)) GA.  The other nine have no validated reference policy (see
:data:`dacbench.envs.env_utils.ga_baselines.UNVALIDATED_COMBINATIONS`) and are
documented in ``docs/source/benchmark_docs/ga.rst``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dacbench.benchmarks import GABenchmark
from dacbench.envs.env_utils import ga_baselines
from dacbench.envs.env_utils.ga_baselines import (
    FAILED_VALIDATION,
    STRUCTURAL,
    UNVALIDATED_COMBINATIONS,
    best_static,
    best_static_portfolio,
    is_validated,
)
from dacbench.envs.theory import OneMax

INSTANCE_DIR = Path(__file__).resolve().parents[2] / "dacbench" / "instance_sets" / "ga"
SCRIPT_PATH = INSTANCE_DIR / "derive_optimal_policy.py"

COLUMNS = ["f_x", "action", "choice", "mean_evals", "n_seeds", "ci_low", "ci_high"]
PORTFOLIO = [1, 2, 4, 8, 16]
PROBLEMS = {"onemax": "OneMax", "leadingones": "LeadingOne"}
ALGORITHMS = ["one_plus_one", "one_plus_lambda", "one_plus_lambda_lambda"]
SIZES = [50, 100, 200]

ALL_COMBINATIONS = [
    (problem, n, algorithm)
    for problem in PROBLEMS
    for n in SIZES
    for algorithm in ALGORITHMS
]

#: The combinations whose derived table ships (9 of the 18).
SHIPPED = [
    ("onemax", 50, "one_plus_one"),
    ("onemax", 50, "one_plus_lambda"),
    ("onemax", 100, "one_plus_one"),
    ("onemax", 100, "one_plus_lambda_lambda"),
    ("onemax", 200, "one_plus_one"),
    ("onemax", 200, "one_plus_lambda"),
    ("onemax", 200, "one_plus_lambda_lambda"),
    ("leadingones", 100, "one_plus_one"),
    ("leadingones", 100, "one_plus_lambda_lambda"),
]

#: The combinations without a validated reference policy (9 of the 18): no CSV
#: ships for them and ``best_static`` raises for them.
UNSHIPPED = [
    combination for combination in ALL_COMBINATIONS if combination not in SHIPPED
]

#: Number of full episodes per policy in the paired dominance test.  These are
#: exactly the 12 seeds of protocol A, which selected the shipped tables (see
#: ``docs/source/benchmark_docs/ga.rst``), so the test reproduces the documented
#: ratios instead of drawing new noise.  The seeds are therefore pinned, not
#: drawn.
VALIDATION_SEEDS = list(range(12))
#: The derived policy may be this much worse than the best fixed action.  This is
#: the tolerance of the documented protocol, not a threshold chosen for the test.
DOMINANCE_TOLERANCE = 1.05

#: Seeds of the held-out drift check, disjoint from the seeds of the protocol
#: that selected the tables (0-11) and from the held-out validation seeds
#: (1000-1011).  Five episodes cannot validate anything; they only detect drift.
DRIFT_SEEDS = list(range(3000, 3005))
#: Looser than the validation tolerance, because five episodes are noisy.
DRIFT_TOLERANCE = 1.15


def policy_path(problem, n, algorithm):
    """Return the path of the shipped policy table of one combination."""
    return INSTANCE_DIR / f"optimal_policy_{problem}_{n}_{algorithm}.csv"


def load_policy(problem, n, algorithm):
    """Load one shipped policy table."""
    return pd.read_csv(policy_path(problem, n, algorithm))


def best_action_per_bucket(table):
    """Return a dict bucket -> action index with the smallest ``mean_evals``."""
    best = table.loc[table.groupby("f_x")["mean_evals"].idxmin()]
    return dict(zip(best["f_x"].astype(int), best["action"].astype(int), strict=True))


def cost_to_go(table):
    """Return the derived cost-to-go per bucket as an array, ``V`` indexed by f(x)."""
    values = table.groupby("f_x")["mean_evals"].min()
    return values.to_numpy()


def make_env(problem, n, algorithm, portfolio):
    """Create a test-mode GA environment on the shipped instance ladder."""
    return GABenchmark(
        config={
            "problem": PROBLEMS[problem],
            "algorithm": algorithm,
            "action_choices": list(portfolio),
            # absolute: GABenchmark prefers a same-named CSV in the current
            # working directory, so a relative name is not reproducible
            "instance_set_path": str(INSTANCE_DIR / f"{problem}_{n}.csv"),
            "cutoff": 10 * n * n,
        }
    ).get_environment(test_env=True)


def fixed_action_policy(index):
    """Return a policy that always plays the given action index."""

    def policy(env, state):
        return index

    policy.reset = lambda: None
    return policy


def run_episodes(env, policy, seeds, budget=None):
    """Run one episode per seed and return the evaluations of every episode.

    The environment progresses through the instance ladder round-robin.  Every
    policy gets its own fresh environment, so all policies see the same
    (seed, instance) pairs and the comparison is paired.

    If ``budget`` is given and the episodes need more than ``budget``
    evaluations together, the run is aborted and ``inf`` is returned.  This is
    exact for the minimum over actions: the mean of an action that already
    spent ``budget`` is above ``budget / len(seeds)``, so the action cannot be
    the cheapest one as long as ``len(seeds) * best * tolerance`` is used (see
    :func:`fixed_action_means`).
    """
    evals = []
    spent = 0
    for seed in seeds:
        policy.reset()
        env.reset(seed=seed)
        while True:
            action = policy(env, None)
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
            if budget is not None and spent + env.total_evals > budget:
                return np.full(len(seeds), np.inf)
        spent += int(env.total_evals)
        evals.append(int(env.total_evals))
    return np.asarray(evals, dtype=float)


def fixed_action_means(problem, n, algorithm, portfolio, seeds):
    """Return ``{portfolio value: mean evaluations}`` for the fixed actions.

    Actions are evaluated in portfolio order.  Once an action needs more than
    ``len(seeds) * best * DOMINANCE_TOLERANCE`` evaluations in total (``best``
    is the best mean so far), it is reported as ``inf``: its mean is then above
    ``best * DOMINANCE_TOLERANCE``, so it cannot be the cheapest fixed action,
    and the minimum over the reported means is the exact minimum over all
    fixed actions.  This keeps the tests fast: the large portfolio values (8,
    16) do not solve the instance and would otherwise burn the full evaluation
    budget of ``10 n^2`` in every episode.
    """
    means = {}
    best = np.inf
    for index, value in enumerate(portfolio):
        limit = (
            None if not np.isfinite(best) else len(seeds) * best * DOMINANCE_TOLERANCE
        )
        values = run_episodes(
            make_env(problem, n, algorithm, portfolio),
            fixed_action_policy(index),
            seeds,
            budget=limit,
        )
        means[value] = float(values.mean())
        best = min(best, means[value])
    return means


def paired_ratio(problem, n, algorithm, seeds):
    """Return (derived mean, best fixed mean, best value, ratio) on paired episodes."""
    portfolio = best_static_portfolio(problem, n, algorithm)
    derived_policy = best_static(
        problem=problem,
        n=n,
        algorithm=algorithm,
        policy_path=policy_path(problem, n, algorithm),
    )
    derived = run_episodes(
        make_env(problem, n, algorithm, portfolio), derived_policy, seeds
    )
    fixed_by_action = fixed_action_means(problem, n, algorithm, portfolio, seeds)
    best_value = min(fixed_by_action, key=fixed_by_action.get)
    ratio = derived.mean() / fixed_by_action[best_value]
    return derived, fixed_by_action[best_value], best_value, ratio


@pytest.fixture(scope="module")
def derivation():
    """Import the derivation script as a module (it is not part of the package)."""
    spec = importlib.util.spec_from_file_location(
        "derive_optimal_policy_for_tests", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shipped_set_is_exactly_the_validated_combinations():
    """The directory holds exactly the 9 validated tables, no stale extras."""
    shipped = sorted(path.name for path in INSTANCE_DIR.glob("optimal_policy_*.csv"))
    expected = sorted(
        f"optimal_policy_{problem}_{n}_{algorithm}.csv"
        for problem, n, algorithm in SHIPPED
    )
    assert len(expected) == 9
    assert shipped == expected
    # the excluded combinations really have no file
    for problem, n, algorithm in UNSHIPPED:
        assert not policy_path(problem, n, algorithm).exists()


def test_derivation_scope_matches_the_shipped_set(derivation):
    """The script's default scope and its exclusion list match the shipped files."""
    assert sorted(derivation.SHIPPED_COMBINATIONS) == sorted(SHIPPED)
    assert sorted(derivation.EXCLUDED_COMBINATIONS) == sorted(UNSHIPPED)
    assert sorted(UNVALIDATED_COMBINATIONS) == sorted(UNSHIPPED)
    # one source of truth: the script re-exports the package's data
    assert derivation.EXCLUDED_COMBINATIONS is UNVALIDATED_COMBINATIONS

    args = derivation.parse_args([])
    assert args.include_unvalidated is False
    assert args.sizes == SIZES
    assert args.problems == list(PROBLEMS)
    assert args.rounds == derivation.DEFAULT_ROUNDS > 0
    # the default selection derives exactly the 9 shipped combinations
    default_scope = [
        (problem, n, algorithm)
        for problem in args.problems
        for n in args.sizes
        for algorithm in args.algorithms
        if (problem, n, algorithm) not in derivation.EXCLUDED_COMBINATIONS
    ]
    assert sorted(default_scope) == sorted(SHIPPED)
    # --all derives all 18
    all_scope = [
        (problem, n, algorithm)
        for problem in args.problems
        for n in args.sizes
        for algorithm in args.algorithms
    ]
    assert len(all_scope) == 18
    assert derivation.parse_args(["--all"]).include_unvalidated is True


def test_unvalidated_data_is_classified_and_carries_ratios():
    """Every unshipped combination names its category, ratio and alternative."""
    categories = {FAILED_VALIDATION, STRUCTURAL}
    assert {entry["category"] for entry in UNVALIDATED_COMBINATIONS.values()} <= (
        categories
    )
    for combination, entry in UNVALIDATED_COMBINATIONS.items():
        assert set(entry["ratios"]) == {"A", "B"}
        # the protocol-A ratio exists for every combination
        assert entry["ratios"]["A"] is not None
        assert entry["best_choice"] in PORTFOLIO
        assert entry["reason"]
        problem, n, algorithm = combination
        assert problem in PROBLEMS
        assert n in SIZES
        assert algorithm in ALGORITHMS
        assert not is_validated(problem, n, algorithm)
    # both protocols are paired runs; there is no third seed set
    # every combination that failed the full-episode validation is worse than the
    # best fixed action on the held-out protocol B, and by more than the tolerance
    # on at least one protocol
    for combination, entry in UNVALIDATED_COMBINATIONS.items():
        if entry["category"] != FAILED_VALIDATION:
            continue
        assert entry["ratios"]["B"] is not None, combination
        assert entry["ratios"]["B"] > DOMINANCE_TOLERANCE, combination
        assert entry["ratios"]["B"] > entry["ratios"]["A"]
    # per-combination best fixed action
    best_choices = {
        combination: entry["best_choice"]
        for combination, entry in UNVALIDATED_COMBINATIONS.items()
    }
    assert best_choices[("leadingones", 50, "one_plus_lambda")] == 2
    assert [c for c, v in best_choices.items() if v == 2] == [
        ("leadingones", 50, "one_plus_lambda"),
        ("leadingones", 100, "one_plus_lambda"),
    ]


@pytest.mark.parametrize(("problem", "n", "algorithm"), SHIPPED)
def test_shipped_policy_is_consistent(problem, n, algorithm):
    """The shipped tables are well formed and satisfy the DP's defining property."""
    table = load_policy(problem, n, algorithm)

    assert list(table.columns) == COLUMNS
    # one row per bucket and action, buckets 0..n (n is the terminal bucket)
    assert len(table) == (n + 1) * len(PORTFOLIO)
    assert sorted(table["f_x"].unique().tolist()) == list(range(n + 1))

    for bucket, group in table.groupby("f_x"):
        bucket = int(bucket)
        assert group["action"].tolist() == list(range(len(PORTFOLIO)))
        assert group["choice"].tolist() == PORTFOLIO
        if bucket == n:
            # nothing left to do at the optimum
            assert (group["mean_evals"] == 0.0).all()
            assert (group["n_seeds"] == 0).all()
            continue
        assert (group["n_seeds"] > 0).all()
        finite = np.isfinite(group["mean_evals"])
        assert finite.any(), f"bucket {bucket} has no finite action value"
        assert (group.loc[finite, "mean_evals"] > 0).all(), bucket
        assert (group.loc[finite, "ci_low"] <= group.loc[finite, "ci_high"]).all()
        # rows without a single simulated improvement are inf in all three
        # numeric columns
        assert np.isinf(group.loc[~finite, "ci_low"]).all()
        assert np.isinf(group.loc[~finite, "ci_high"]).all()

    # the DP's defining property: the derived best action has the smallest
    # mean_evals of its bucket (there is no extra "is best" column)
    best = best_action_per_bucket(table)
    for bucket, action in best.items():
        row = table[(table["f_x"] == bucket) & (table["action"] == action)]
        assert (
            row["mean_evals"].iloc[0]
            == table[table["f_x"] == bucket]["mean_evals"].min()
        )

    # the cost-to-go decreases towards the optimum.  It is not required to be
    # monotone bucket by bucket (the buckets are estimated independently and the
    # policy may jump between buckets), but the trend must be clearly decreasing
    # over windows of ten buckets
    values = cost_to_go(table)
    assert values[n] == 0.0
    assert np.all(values[: n - 10] >= values[10:n])
    assert values[n // 2] > values[n - 1]
    assert values[0] > 1.5 * values[n - 1]


@pytest.mark.parametrize(("problem", "n", "algorithm"), SHIPPED)
def test_shipped_policy_dominates_best_fixed_action(problem, n, algorithm):
    """Regression/drift test: the shipped policy still beats the best fixed action.

    This test is **pinned to the documented protocol** and is not a validation:
    it reproduces protocol A of ``docs/source/benchmark_docs/ga.rst`` (12 paired
    episodes, the shipped instance ladder, cutoff ``10 n^2``, tolerance 1.05) on
    the shipped tables, so a change of the tables, of the operators or of the
    environment that pushes a shipped policy above the tolerance fails here.
    The seeds are pinned for exactly that reason: with fewer seeds a
    combination can flip over the tolerance for purely statistical reasons.

    Both policy kinds run on the same (seed, instance) pairs, so the comparison
    is paired.  The measured ratios of the shipped tables are 0.84-0.99 on A and
    0.79-1.01 on B (see the docs page).
    """
    derived, best_fixed, best_value, ratio = paired_ratio(
        problem, n, algorithm, VALIDATION_SEEDS
    )
    assert ratio <= DOMINANCE_TOLERANCE, (
        f"{problem} n={n} {algorithm}: the derived policy needs "
        f"{derived.mean():.1f} evaluations, but the fixed action "
        f"{best_value} needs {best_fixed:.1f} (ratio {ratio:.3f})"
    )


@pytest.mark.parametrize(("problem", "n", "algorithm"), SHIPPED)
def test_held_out_drift_check(problem, n, algorithm):
    """Drift check on held-out seeds 3000-3004 (5 episodes, 15% tolerance).

    This is a **drift check, not a validation claim**: five episodes are far too
    few to validate a policy (the validation in the docs uses the 12 seeds of A
    and the 12 held-out seeds of B).  It only detects that the shipped policy has
    moved far away
    from the best fixed action since the measurement.
    """
    _derived, best_fixed, best_value, ratio = paired_ratio(
        problem, n, algorithm, DRIFT_SEEDS
    )
    assert ratio <= DRIFT_TOLERANCE, (
        f"{problem} n={n} {algorithm}: drift check failed, the shipped policy "
        f"needs {ratio:.3f}x the evaluations of the best fixed action "
        f"{best_value} ({best_fixed:.1f}) on held-out seeds {DRIFT_SEEDS}"
    )


@pytest.mark.parametrize(("problem", "n", "algorithm"), UNSHIPPED)
def test_best_static_raises_for_unvalidated_combinations(problem, n, algorithm):
    """Unshipped combinations raise a clear, actionable ValueError."""
    with pytest.raises(ValueError, match="no validated reference table exists"):
        best_static(problem=problem, n=n, algorithm=algorithm)


def test_unvalidated_message_names_the_alternative():
    """The error explains the cause and what to use instead."""
    with pytest.raises(ValueError, match="no validated reference table") as excinfo:
        best_static(problem="LeadingOne", n=200, algorithm="one_plus_lambda_lambda")
    message = str(excinfo.value)
    assert "suffix" in message  # why the bucket abstraction fails
    assert "success_rule" in message  # what to use instead
    assert "action index 0" in message  # the measured best fixed action
    assert "1.269" in message  # the measured ratios of this combination
    assert "seeds 0-11" in message  # the selection protocol
    assert "seeds 1000-1011" in message  # the held-out protocol
    assert "2000" not in message  # the unpaired run is gone
    assert "--all" in message
    assert "policy_path" in message


def test_unvalidated_message_names_the_best_fixed_action_per_combination():
    """The error names the best fixed action measured for that combination.

    This is not one global default: the best fixed action is measured per
    combination (lambda = 2 for both LeadingOne (1+lambda) EAs, action index 0
    everywhere else).
    """
    for (problem, n, algorithm), entry in UNVALIDATED_COMBINATIONS.items():
        index = PORTFOLIO.index(entry["best_choice"])
        with pytest.raises(ValueError, match=f"action index {index}"):
            best_static(problem=problem, n=n, algorithm=algorithm)


def test_best_static_accepts_explicit_unvalidated_table(tmp_path):
    """An explicitly passed table of an unshipped combination is usable."""
    source = INSTANCE_DIR / "optimal_policy_leadingones_100_one_plus_one.csv"
    path = tmp_path / "optimal_policy_leadingones_200_one_plus_lambda.csv"
    path.write_bytes(source.read_bytes())
    policy = best_static(policy_path=path)
    env = make_env("leadingones", 200, "one_plus_lambda", PORTFOLIO)
    env.reset(seed=0)
    assert 0 <= policy(env, None) < len(PORTFOLIO)


@pytest.mark.parametrize(("problem", "n", "algorithm"), SHIPPED)
def test_derived_policy_varies_with_fitness(problem, n, algorithm):
    """The derived choice is not constant over the fitness buckets."""
    best = best_action_per_bucket(load_policy(problem, n, algorithm))
    distinct = sorted(set(best.values()))
    assert len(distinct) >= 2, (
        f"{problem} n={n} {algorithm}: the derived action is always "
        f"{distinct[0]}, the policy does not react to f(x)"
    )


def test_tiny_derivation_is_reproducible(derivation, tmp_path):
    """The identical invocation produces bit-identical CSVs."""
    outputs = []
    for run in range(2):
        out_dir = tmp_path / f"run{run}"
        derivation.main(
            [
                "--problems",
                "onemax",
                "--sizes",
                "50",
                "--algorithms",
                "one_plus_one",
                "--n-seeds",
                "10",
                "--rounds",
                "1",
                "--episodes-per-round",
                "4",
                "--out-dir",
                str(out_dir),
            ]
        )
        path = out_dir / "optimal_policy_onemax_50_one_plus_one.csv"
        outputs.append(path.read_bytes())

    assert outputs[0] == outputs[1]
    table = pd.read_csv(tmp_path / "run0" / path.name)
    assert list(table.columns) == COLUMNS
    assert len(table) == 51 * len(PORTFOLIO)


def test_derivation_skips_unvalidated_combinations(derivation, tmp_path):
    """``main`` writes nothing for the excluded combinations without ``--all``."""
    derivation.main(
        [
            "--problems",
            "leadingones",
            "--sizes",
            "50",
            "--algorithms",
            "one_plus_one",
            "--n-seeds",
            "10",
            "--rounds",
            "0",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert not list(tmp_path.glob("*.csv"))


def test_derivation_reports_convergence(derivation, tmp_path):
    """``--rounds`` rounds of a stable policy converge and are reported as such."""
    out_dir = tmp_path / "converged"
    derivation.main(
        [
            "--problems",
            "onemax",
            "--sizes",
            "50",
            "--algorithms",
            "one_plus_one",
            "--n-seeds",
            "10",
            "--rounds",
            "0",
            "--out-dir",
            str(out_dir),
        ]
    )
    # --rounds 0 stops after the fresh-state round: nothing can have converged
    _table, stats = derivation.derive_optimal_policy(
        "onemax", 50, "one_plus_one", PORTFOLIO, 10, 0, rounds=0
    )
    assert stats["converged"] is False
    assert stats["n_rounds"] == 0
    # a full pass with a stable policy converges after one on-policy round
    _table, stats = derivation.derive_optimal_policy(
        "onemax", 50, "one_plus_one", PORTFOLIO, 10, 0, rounds=2, episodes_per_round=2
    )
    assert stats["n_rounds"] <= 2
    assert stats["converged"] is (stats["rounds"][-1]["changes"] == 0)


class TestEstimatorInternals:
    """Unit tests of the estimation internals (no environment, no episodes)."""

    def test_self_loop_correction_is_analytic(self, derivation):
        """``V = sum(e)/q + E[V(b')|improve]`` is checked against closed form.

        The division by ``q`` is the whole point of the estimator: without it the
        generations that do not improve (whose successor bucket is the bucket
        itself, with unknown value) would be counted with a cost of zero.
        """
        values = np.zeros(11)
        values[5] = np.inf  # the bucket itself is what is being computed
        values[6] = 10.0
        # one action, four samples from bucket 5; the first two improve to 6
        evals = [[1.0, 2.0, 3.0, 4.0]]
        succ = [[6, 6, 5, 5]]

        means, counts = derivation._estimate_bucket_means(evals, succ, 5, values)

        assert counts.tolist() == [2]
        # sum(e) / q + V(6) = 10 / 2 + 10 = 15.  Without the /q correction the
        # answer would be mean(e) + 10 = 12.5, and averaging only the improving
        # samples would give 1.5 + 10 = 11.5.
        assert means[0] == pytest.approx(15.0)

        # self-loops only: no improvement, hence no finite estimate
        means, counts = derivation._estimate_bucket_means(
            [[1.0, 1.0]], [[5, 5]], 5, values
        )
        assert counts.tolist() == [0]
        assert np.isinf(means[0])

    def test_self_loop_correction_with_a_success_probability_below_one(
        self, derivation
    ):
        """A single improvement in N samples multiplies the expected cost by N."""
        values = np.zeros(11)
        values[6] = 0.0
        n_samples = 50
        evals = [[2.0] * n_samples]
        succ = [[6] + [5] * (n_samples - 1)]

        means, counts = derivation._estimate_bucket_means(evals, succ, 5, values)

        assert counts.tolist() == [1]
        # sum(e) / q = (2 * 50) / 1 = 100, and V(6) = 0
        assert means[0] == pytest.approx(100.0)

    @pytest.mark.parametrize(("problem", "n", "algorithm"), SHIPPED)
    def test_confidence_interval_brackets_the_estimate(self, problem, n, algorithm):
        """``ci_low <= mean_evals <= ci_high`` on every row of every shipped table."""
        table = load_policy(problem, n, algorithm)
        rows = table[(table["f_x"] < n) & np.isfinite(table["mean_evals"])]
        assert len(rows) > 0
        tolerance = 1e-9 * rows["mean_evals"].abs().clip(lower=1.0)
        assert (rows["ci_low"] <= rows["mean_evals"] + tolerance).all()
        assert (rows["mean_evals"] <= rows["ci_high"] + tolerance).all()

    def test_run_policy_pass_is_deterministic(self, derivation):
        """Two identical passes produce identical rows (fixed RNG streams)."""
        portfolio = (1, 2)
        n, n_seeds = 6, 50
        env = derivation.make_generation_env("OneMax", "one_plus_one", portfolio)
        env.n = n
        args = (env, OneMax, n, "one_plus_one", portfolio, n_seeds, 0, (0, 0, 0), None)

        values_a, policy_a, rows_a, stats_a = derivation.run_policy_pass(*args)
        values_b, policy_b, rows_b, stats_b = derivation.run_policy_pass(*args)

        assert rows_a == rows_b
        assert policy_a == policy_b
        assert np.array_equal(values_a, values_b)
        assert stats_a == stats_b
        # the pass produced one row per (bucket, action) plus the terminal bucket
        assert len(rows_a) == n * len(portfolio) + len(portfolio)
        assert all(row["f_x"] in range(n + 1) for row in rows_a)


def test_baseline_module_uses_the_same_data():
    """The package's unvalidated data and the shipped files agree."""
    for problem, n, algorithm in SHIPPED:
        assert is_validated(problem, n, algorithm)
        assert policy_path(problem, n, algorithm).is_file()
    for problem, n, algorithm in UNSHIPPED:
        assert not is_validated(problem, n, algorithm)
        assert (
            ga_baselines.policy_file_name(PROBLEMS[problem], n, algorithm)
            == policy_path(problem, n, algorithm).name
        )
