"""Tests for the exact optimal bucket policies of the GA benchmark.

The tables ``dacbench/instance_sets/ga/optimal_policy_onemax_*_{one_plus_one,
one_plus_lambda}.csv`` and ``exact_fixed_action_counts.csv`` are produced by
``dacbench/instance_sets/ga/exact_optimal_policy.py`` (a standalone script, see
its module docstring).  They hold the *exact* optimal expected number of
evaluations of the fitness-bucket model, which for OneMax with the (1+1) EA and
the (1+lambda) EA is the true optimum: ``f(x) = sum_i x_i`` is symmetric, so the
number of ones is a sufficient statistic and the successor distribution of a
generation is known in closed form.

What is checked here, in order of importance:

1. the analytic transition matrix against the environment itself
   (:meth:`TestTransitionMatrix.test_matches_environment_generations`, total
   variation distance over 20 000 generations per row) and against an exhaustive
   enumeration of all bitstrings and all flip sets at n = 6, 7, 8;
2. the exact evaluation cost of a generation (1 for the (1+1) EA, ``lambda`` for
   the (1+lambda) EA);
3. the dynamic programme against full benchmark episodes;
4. the contract of the shipped CSVs, the fixed-action costs, the refusals for
   combinations where the bucket model is not exact, and determinism.

Most tests are fast.  The two marked ``slow`` (full episodes at n = 20 and
n = 50) dominate the runtime of this file and use a Monte-Carlo tolerance of
four sample standard errors plus 2 % (see :func:`assert_matches_prediction`).
"""

from __future__ import annotations

import importlib.util
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dacbench.benchmarks import GABenchmark
from dacbench.envs.env_utils.ga_baselines import best_static, load_policy_table
from dacbench.envs.theory import OneMax

INSTANCE_DIR = Path(__file__).resolve().parents[2] / "dacbench" / "instance_sets" / "ga"
SCRIPT_PATH = INSTANCE_DIR / "exact_optimal_policy.py"

PORTFOLIO = [1, 2, 4, 8, 16]
#: Portfolio for the small sizes, where ``16 > n`` (the environment rejects
#: action values above ``n``).
SMALL_PORTFOLIO = [1, 2, 4]
SIZES = [50, 100, 200]
ALGORITHMS = ["one_plus_one", "one_plus_lambda"]
COLUMNS = [
    "f_x",
    "action",
    "choice",
    "mean_evals",
    "n_seeds",
    "ci_low",
    "ci_high",
    "exact",
]
FIXED_COLUMNS = ["n", "algorithm", "start_f", "action", "choice", "expected_evals"]

#: Number of generations per (bucket, action) of the model validation.  With
#: 20 000 samples the expected total variation distance between the empirical
#: and the analytic bucket distribution is about ``sqrt(K / (2 pi N)) ~ 0.01``
#: for the ``K <= 21`` buckets of n = 20, so ``TV_TOLERANCE`` is more than five
#: standard deviations above the expectation and does not flake.
TV_SAMPLES = 20_000
TV_TOLERANCE = 0.05

#: Full episodes of the DP-vs-episodes tests.  Hitting times are strongly
#: right-skewed, so the tolerance is ``MC_SIGMAS`` sample standard errors plus
#: ``MC_RELATIVE`` of the predicted value instead of a plain z-interval.
EPISODES_N20 = 200
EPISODES_N50 = 100
MC_SIGMAS = 4.0
MC_RELATIVE = 0.02


@pytest.fixture(scope="module")
def exact():
    """Import the exact-derivation script as a module (not part of the package)."""
    spec = importlib.util.spec_from_file_location(
        "exact_optimal_policy_for_tests", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def policy_table_path(n, algorithm):
    """Return the path of the shipped exact table of one combination."""
    return INSTANCE_DIR / f"optimal_policy_onemax_{n}_{algorithm}.csv"


def make_env(problem, n, algorithm, portfolio, test_env=True):
    """Create a GA environment on the shipped instance ladder of size ``n``."""
    benchmark = GABenchmark(
        config={
            "problem": problem,
            "algorithm": algorithm,
            "action_choices": list(portfolio),
            # absolute: GABenchmark prefers a same-named CSV in the current
            # working directory, so a relative name is not reproducible
            "instance_set_path": str(INSTANCE_DIR / f"onemax_{n}.csv"),
            "cutoff": 10 * n * n,
        }
    )
    env = benchmark.get_environment(test_env=test_env)
    if not test_env:
        # a one-generation simulator: the state is assigned per generation
        env.n = n
    return env


def brute_force_rows(n, p, exact):
    """Enumerate every bitstring and flip set and group the rows by fitness.

    This is the independent check of the hypergeometric argument: it never uses
    the closed form of ``P(A = a | f, l)``, it flips bits.

    Returns:
    -------
    dict
        ``{f: [row, ...]}`` -- one row per bitstring with ``f`` ones.

    """
    log_fact = exact.log_factorials(n)
    flip_pmf = exact.flip_count_distribution(n, p, log_fact)
    rows = {}
    for bits in itertools.product([0, 1], repeat=n):
        string = np.array(bits, dtype=bool)
        f = int(string.sum())
        row = np.zeros(n + 1)
        for index, flips in enumerate(range(1, n + 1)):
            choices = list(itertools.combinations(range(n), flips))
            for locs in choices:
                child = string.copy()
                child[list(locs)] = ~child[list(locs)]
                row[max(f, int(child.sum()))] += flip_pmf[index] / len(choices)
        rows.setdefault(f, []).append(row)
    return rows


def optimal_actions(table):
    """Return ``{bucket: action index}`` of the smallest ``mean_evals`` per bucket."""
    return {
        int(f_x): int(group.loc[group["mean_evals"].idxmin(), "action"])
        for f_x, group in table.groupby("f_x")
    }


def build_policy(exact, n, algorithm):
    """Return ``{bucket: action index}`` of the exact optimal policy.

    The policy is read from the shipped CSV when there is one (n in
    ``SIZES``), so that the episode tests also cover the file on disk, and from
    the script's in-memory table otherwise (n = 20 has no shipped file).
    """
    path = policy_table_path(n, algorithm)
    if path.is_file():
        return optimal_actions(load_policy_table(path))
    return optimal_actions(exact.policy_table(n, algorithm, PORTFOLIO))


def run_policy_episodes(exact, n, algorithm, action, episodes, use_fixed):
    """Run full episodes of one policy and return (evaluations, prediction).

    Every policy gets its own fresh environment: ``AbstractEnv`` progresses
    through the instance ladder round-robin, so reusing an environment across
    policies would pair a policy with the instances of the previous policy.  The
    prediction of episode ``k`` is ``1 + W[action, f(x_0)]`` (fixed action) or
    ``1 + V[f(x_0)]`` (optimal policy): the environment counts the evaluation of
    the initial individual, the DP does not.
    """
    solution = exact.solve(n, algorithm, PORTFOLIO)
    policy = build_policy(exact, n, algorithm) if not use_fixed else None
    env = make_env("OneMax", n, algorithm, PORTFOLIO)
    evals = []
    predictions = []
    for seed in range(episodes):
        env.reset(seed=seed)
        start = int(env.init_obj)
        if use_fixed:
            predictions.append(1.0 + solution["fixed"][action, start])
        else:
            predictions.append(1.0 + solution["values"][start])
        while True:
            chosen = action if use_fixed else policy[int(env.x.fitness)]
            _, _, terminated, truncated, _ = env.step(chosen)
            if terminated or truncated:
                break
        evals.append(int(env.total_evals))
    return np.asarray(evals, dtype=float), np.asarray(predictions, dtype=float)


def assert_matches_prediction(samples, predictions, label):
    """Assert that the mean of ``samples`` matches the exact ``predictions``.

    Nothing is cut off by the evaluation budget (the instance ladder is solvable
    for the tested policies), so every episode terminates at the optimum.
    """
    observed = samples.mean()
    predicted = predictions.mean()
    standard_error = samples.std(ddof=1) / math.sqrt(samples.size)
    tolerance = MC_SIGMAS * standard_error + MC_RELATIVE * predicted
    assert abs(observed - predicted) <= tolerance, (
        f"{label}: {samples.size} episodes spent {observed:.2f} +- "
        f"{standard_error:.2f} evaluations, the exact model predicts "
        f"{predicted:.2f} (tolerance {tolerance:.2f})"
    )


class TestTransitionMatrix:
    """The analytic bucket model against exhaustive enumeration and the env."""

    @pytest.mark.parametrize("n", [6, 7, 8])
    @pytest.mark.parametrize("rate", ["one_over_n", "two_over_n", "half"])
    def test_matches_exhaustive_enumeration(self, exact, n, rate):
        """Every bitstring-level generation reproduces the analytic row.

        The test does two things at once: it validates ``T`` against a brute-force
        enumeration of all ``2**n`` bitstrings and all flip sets (this is the
        check that proves the hypergeometric argument), and it shows that all
        bitstrings with the same ``f(x)`` share one row -- the sufficiency of the
        fitness bucket that the whole derivation rests on.
        """
        p = {"one_over_n": 1.0 / n, "two_over_n": 2.0 / n, "half": 0.5}[rate]
        analytic = exact.single_child_rows(n, p, exact.log_factorials(n))
        rows = brute_force_rows(n, p, exact)

        assert sorted(rows) == list(range(n + 1))
        for f, per_string in rows.items():
            for row in per_string:
                # identical for every bitstring with f(x) = f: no state dependence
                assert np.allclose(row, analytic[f], atol=1e-12), f
            assert np.allclose(np.mean(per_string, axis=0), analytic[f], atol=1e-12)
        assert np.allclose(analytic.sum(axis=1), 1.0, atol=1e-12)
        # elitist acceptance: no mass below the parent's fitness
        assert np.allclose(np.tril(analytic[:n, :n], -1), 0.0)

    @pytest.mark.parametrize("n", [6, 8])
    def test_flip_count_distribution_matches_math_comb(self, exact, n):
        """``P(L = l)`` agrees with an exact integer-combinatorics evaluation."""
        log_fact = exact.log_factorials(n)
        for p in (1.0 / n, 2.0 / n, 0.5, 1.0):
            pmf = exact.flip_count_distribution(n, p, log_fact)
            if p == 1.0:
                # every bit flips: L = n with probability 1 (no 0 * log(0))
                assert pmf.tolist() == [0.0] * (n - 1) + [1.0]
                continue
            total = 1.0 - (1.0 - p) ** n
            expected = np.array(
                [
                    math.comb(n, length) * p**length * (1.0 - p) ** (n - length) / total
                    for length in range(1, n + 1)
                ]
            )
            assert np.allclose(pmf, expected, atol=1e-15, rtol=1e-12)

    def test_one_plus_lambda_is_the_max_of_iid_children(self, exact):
        """The ``lambda`` row is the max of ``lambda`` iid single-child rows.

        The single-child row is taken from the *brute-force* enumeration, so the
        cumulative-power formula is checked independently of the closed form.
        """
        n, lam = 6, 3
        p = 1.0 / n
        rows = brute_force_rows(n, p, exact)
        matrices = exact.transition_matrices(n, "one_plus_lambda", [lam])
        for f in range(n):
            cumulative = np.cumsum(np.mean(rows[f], axis=0))
            expected = np.zeros(n + 1)
            expected[f] = cumulative[f] ** lam
            for g in range(f + 1, n + 1):
                expected[g] = cumulative[g] ** lam - cumulative[g - 1] ** lam
            assert np.allclose(matrices[0, f], expected, atol=1e-12), f
        # lambda = 1 is the (1+1) EA with r = 1, row by row
        one = exact.single_child_rows(n, p, exact.log_factorials(n))
        single_lambda = exact.transition_matrices(n, "one_plus_lambda", [1])
        assert np.allclose(single_lambda[0], one, atol=1e-16)

    def test_rejects_action_values_above_n(self, exact):
        """Action values outside ``[1, n]`` have no bucket-model meaning."""
        with pytest.raises(ValueError, match=r"must be in \[1, 6\]"):
            exact.transition_matrices(6, "one_plus_one", [1, 16])

    @pytest.mark.parametrize(
        ("algorithm", "value"),
        [
            ("one_plus_one", 1),
            ("one_plus_one", 4),
            ("one_plus_one", 16),
            ("one_plus_lambda", 1),
            ("one_plus_lambda", 4),
            ("one_plus_lambda", 16),
        ],
    )
    @pytest.mark.parametrize("bucket", [0, 10, 19])
    def test_matches_environment_generations(self, exact, algorithm, value, bucket):
        """Empirical generations of ``GAEnv._generation`` reproduce ``T``.

        ``TV_SAMPLES = 20 000`` generations are drawn from the environment's own
        RNG (one fresh bitstring with ``f(x) = bucket`` per generation, so the
        samples are iid) and the total variation distance to the analytic row
        must stay below ``TV_TOLERANCE = 0.05``.  The expected distance at
        20 000 samples is about 0.01 for the K <= 21 buckets of n = 20, so the
        tolerance is more than five standard deviations away from it.
        """
        n = 20
        rng = np.random.default_rng(12345)
        env = make_env("OneMax", n, algorithm, PORTFOLIO, test_env=False)
        env.np_random = rng
        row = exact.transition_matrices(n, algorithm, [value])[0, bucket]
        counts = np.zeros(n + 1)
        for _ in range(TV_SAMPLES):
            env.x = OneMax(n=n, rng=rng, initObj=bucket)
            env._generation(value)
            counts[env.x.fitness] += 1
        empirical = counts / counts.sum()
        tv = 0.5 * np.abs(empirical - row).sum()
        assert tv < TV_TOLERANCE, (
            f"{algorithm} value={value} bucket={bucket}: TV distance {tv:.4f} "
            f">= {TV_TOLERANCE} at {TV_SAMPLES} samples"
        )


class TestEvaluationCosts:
    """Every generation costs exactly 1 (one_plus_one) or lambda evaluations."""

    @pytest.mark.parametrize("value", PORTFOLIO)
    def test_one_plus_one_costs_one(self, value):
        """``(1+1) EA_{>0}``: one child, one evaluation, whatever the strength."""
        env = make_env("OneMax", 20, "one_plus_one", PORTFOLIO, test_env=False)
        env.x = OneMax(n=20, rng=env.np_random, initObj=10)
        for _ in range(50):
            assert env._generation(value) == 1

    @pytest.mark.parametrize("value", PORTFOLIO)
    def test_one_plus_lambda_costs_lambda(self, value):
        """``(1+lambda) EA``: every child is evaluated, also the worse ones."""
        env = make_env("OneMax", 20, "one_plus_lambda", PORTFOLIO, test_env=False)
        env.x = OneMax(n=20, rng=env.np_random, initObj=10)
        for _ in range(20):
            assert env._generation(value) == value

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_episode_total_evals(self, exact, algorithm):
        """A full episode spends ``1 + steps * cost`` evaluations.

        The environment counts the initial individual, so ``total_evals - 1`` is
        exactly the number of steps times the per-generation cost of the played
        action.
        """
        assert exact.evaluations_per_generation(
            20, "one_plus_one", PORTFOLIO
        ).tolist() == [1.0] * len(PORTFOLIO)
        assert exact.evaluations_per_generation(
            20, "one_plus_lambda", PORTFOLIO
        ).tolist() == [float(value) for value in PORTFOLIO]

        value = 4
        cost = 1 if algorithm == "one_plus_one" else value
        env = make_env("OneMax", 20, algorithm, PORTFOLIO)
        env.reset(seed=0)
        steps = 0
        while True:
            _, _, terminated, truncated, _ = env.step(PORTFOLIO.index(value))
            steps += 1
            if terminated or truncated:
                break
        assert env.total_evals == 1 + steps * cost


class TestDynamicProgram:
    """The dynamic programme against the value it claims to compute."""

    @pytest.mark.parametrize("n", [6, 7, 8])
    def test_greedy_policy_is_a_fixed_point_of_the_dp(self, exact, n):
        """``V[f] = V[f, argmin]`` and ``V[f] <= W_a[f]`` for every action.

        This is the defining property of the DP: the column of the selected
        action equals the value function, and no constant action can be better
        than the optimal policy, because ``V`` is a minimum over all bucket
        policies of the model.
        """
        for algorithm in ALGORITHMS:
            solution = exact.solve(n, algorithm, SMALL_PORTFOLIO)
            values = solution["values"]
            columns = solution["columns"]
            policy = solution["policy"]
            assert values[n] == 0.0
            for f in range(n):
                assert values[f] == pytest.approx(columns[policy[f], f])
                assert values[f] <= columns[:, f].min() + 1e-12
                assert np.all(values[f] <= solution["fixed"][:, f] + 1e-9)
                assert np.isfinite(values[f])
                assert values[f] > 0

    def test_self_loop_correction(self, exact):
        """The column is ``(c + sum_{g>f} T[f,g] V[g]) / (1 - T[f,f])``."""
        n, algorithm, value = 8, "one_plus_one", 2
        solution = exact.solve(n, algorithm, SMALL_PORTFOLIO)
        index = SMALL_PORTFOLIO.index(value)
        matrices = solution["matrices"]
        values = solution["values"]
        f = 3
        improve = 1.0 - matrices[index, f, f]
        assert improve > 0
        expected = (
            solution["costs"][index] + matrices[index, f, f + 1 :] @ values[f + 1 :]
        ) / improve
        assert solution["columns"][index, f] == pytest.approx(expected)

    @pytest.mark.slow
    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_matches_full_episodes_at_n20(self, exact, algorithm):
        """The greedy policy and three fixed actions match their exact costs."""
        for action, use_fixed in [(0, False), (0, True), (1, True), (2, True)]:
            samples, predictions = run_policy_episodes(
                exact,
                n=20,
                algorithm=algorithm,
                action=action,
                episodes=EPISODES_N20,
                use_fixed=use_fixed,
            )
            label = (
                f"n=20 {algorithm} optimal policy"
                if not use_fixed
                else f"n=20 {algorithm} fixed action {PORTFOLIO[action]}"
            )
            assert_matches_prediction(samples, predictions, label)

    @pytest.mark.slow
    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_matches_full_episodes_at_n50(self, exact, algorithm):
        """The same check at n = 50, where the hitting times are large."""
        for action, use_fixed in [(0, False), (0, True)]:
            samples, predictions = run_policy_episodes(
                exact,
                n=50,
                algorithm=algorithm,
                action=action,
                episodes=EPISODES_N50,
                use_fixed=use_fixed,
            )
            label = (
                f"n=50 {algorithm} optimal policy"
                if not use_fixed
                else f"n=50 {algorithm} fixed action {PORTFOLIO[action]}"
            )
            # the instances must be solvable well within the budget, otherwise
            # the episode is cut off and the comparison is meaningless
            assert predictions.max() < 10 * 50 * 50 / 4
            assert_matches_prediction(samples, predictions, label)


class TestShippedTables:
    """The six shipped exact tables and the fixed-action cost table."""

    @pytest.mark.parametrize(
        ("n", "algorithm"), list(itertools.product(SIZES, ALGORITHMS))
    )
    def test_table_contract(self, exact, n, algorithm):
        """Columns, rows, markers and the DP values of one shipped table."""
        path = policy_table_path(n, algorithm)
        assert path.is_file()
        table = load_policy_table(path)
        assert list(table.columns) == COLUMNS
        assert len(table) == (n + 1) * len(PORTFOLIO)
        assert sorted(table["f_x"].unique().tolist()) == list(range(n + 1))

        # there is no sampling: the markers say so
        assert (table["exact"] == 1).all()
        assert (table["n_seeds"] == 0).all()
        assert np.array_equal(table["ci_low"], table["mean_evals"])
        assert np.array_equal(table["ci_high"], table["mean_evals"])

        solution = exact.solve(n, algorithm, PORTFOLIO)
        fixed = solution["fixed"]
        for bucket, group in table.groupby("f_x"):
            bucket = int(bucket)
            assert group["action"].tolist() == list(range(len(PORTFOLIO)))
            assert group["choice"].tolist() == PORTFOLIO
            means = group["mean_evals"].to_numpy(dtype=float)
            if bucket == n:
                assert (means == 0.0).all()
                continue
            assert np.isfinite(means).all()
            # the CSV carries the DP values in full precision; a different
            # platform may sum in a different order, hence the tolerance of the
            # --check path (see exact.CHECK_RTOL)
            assert np.allclose(
                means, solution["columns"][:, bucket], rtol=exact.CHECK_RTOL
            )
            # the per-bucket argmin is the DP's optimal action
            assert int(np.argmin(means)) == int(solution["policy"][bucket])
            # playing the action once and then optimally is never worse than
            # always playing it
            assert np.all(means <= fixed[:, bucket] + 1e-9)
            # and the value function is the cost of the selected action
            assert means.min() == pytest.approx(solution["values"][bucket])

    @pytest.mark.parametrize(
        ("n", "algorithm"), list(itertools.product(SIZES, ALGORITHMS))
    )
    def test_value_function_decreases_towards_the_optimum(self, n, algorithm):
        """The exact cost-to-go falls monotonically from bucket 0 to bucket n."""
        table = load_policy_table(policy_table_path(n, algorithm))
        values = table.groupby("f_x")["mean_evals"].min().to_numpy(dtype=float)
        assert values[n] == 0.0
        assert np.all(np.diff(values) <= 0)
        assert values[0] > 1.5 * values[n - 1] > 0

    def test_fixed_action_table_contract(self, exact):
        """The fixed-action table covers every combination and agrees with the DP."""
        path = INSTANCE_DIR / "exact_fixed_action_counts.csv"
        table = pd.read_csv(path)
        assert list(table.columns) == FIXED_COLUMNS
        expected_rows = len(ALGORITHMS) * sum(
            (n + 1) * (len(PORTFOLIO) + 1) for n in SIZES
        )
        assert len(table) == expected_rows
        assert set(table["algorithm"]) == set(ALGORITHMS)
        assert sorted(table["action"].unique().tolist()) == [
            -1,
            *range(len(PORTFOLIO)),
        ]
        assert table[table["action"] == -1]["choice"].eq(0).all()

        for n in SIZES:
            for algorithm in ALGORITHMS:
                solution = exact.solve(n, algorithm, PORTFOLIO)
                block = table[(table["n"] == n) & (table["algorithm"] == algorithm)]
                optimal = block[block["action"] == -1].sort_values("start_f")
                assert np.array_equal(optimal["start_f"].to_numpy(), np.arange(n + 1))
                expected = np.concatenate([solution["values"][:n], [0.0]])
                assert np.allclose(
                    optimal["expected_evals"].to_numpy(dtype=float),
                    expected,
                    rtol=exact.CHECK_RTOL,
                )
                for index, value in enumerate(PORTFOLIO):
                    rows = block[block["action"] == index].sort_values("start_f")
                    assert rows["choice"].eq(value).all()
                    expected = np.concatenate([solution["fixed"][index, :n], [0.0]])
                    assert np.allclose(
                        rows["expected_evals"].to_numpy(dtype=float),
                        expected,
                        rtol=exact.CHECK_RTOL,
                    )

    def test_one_plus_lambda_optimum_is_the_constant_action(self, exact):
        """For the (1+lambda) EA on OneMax the optimal bucket policy is lambda = 1.

        A documented finding, not an assumption: every bucket of every shipped
        size selects the first portfolio entry, and the exact value equals the
        cost of always playing ``lambda = 1``.  The mutation rate is fixed at
        ``1/n``, so a larger population multiplies the cost of a generation and
        the number of children by the same factor and cannot reduce the expected
        number of evaluations in this model.
        """
        for n in SIZES:
            solution = exact.solve(n, "one_plus_lambda", PORTFOLIO)
            assert set(solution["policy"][:n].tolist()) == {0}
            assert solution["values"][0] == pytest.approx(
                solution["fixed"][0, 0], rel=1e-12
            )
            # the gain of the (1+1) EA is real: it does beat its best fixed action
            one_plus_one = exact.solve(n, "one_plus_one", PORTFOLIO)
            assert one_plus_one["values"][0] < 0.95 * one_plus_one["fixed"][:, 0].min()


class TestRefusals:
    """Combinations for which the bucket model is not exact are refused."""

    @pytest.mark.parametrize(
        ("problem", "algorithm", "needle"),
        [
            ("onemax", "one_plus_lambda_lambda", "Hamming distance"),
            ("leadingones", "one_plus_one", "suffix"),
            ("leadingones", "one_plus_lambda", "suffix"),
            ("leadingones", "one_plus_lambda_lambda", "suffix"),
        ],
    )
    def test_require_exact_raises(self, exact, problem, algorithm, needle):
        """``require_exact`` raises a ``ValueError`` that names the reason."""
        with pytest.raises(exact.NotExactError, match=needle) as excinfo:
            exact.require_exact(problem, algorithm)
        message = str(excinfo.value)
        assert problem in message
        assert algorithm in message
        assert "derive_optimal_policy.py" in message  # the alternative
        assert isinstance(excinfo.value, ValueError)

    def test_exact_combinations_pass(self, exact):
        """The six exact combinations are accepted, unknown ones are not."""
        for algorithm in ALGORITHMS:
            assert exact.require_exact("onemax", algorithm) is None
        with pytest.raises(ValueError, match="unknown problem"):
            exact.require_exact("trap", "one_plus_one")
        with pytest.raises(ValueError, match="unknown algorithm"):
            exact.require_exact("onemax", "two_plus_two")

    @pytest.mark.parametrize(
        "argv",
        [
            ["--algorithms", "one_plus_lambda_lambda"],
            ["--problems", "leadingones"],
            ["--problems", "leadingones", "--algorithms", "one_plus_one"],
        ],
    )
    def test_cli_refuses(self, exact, tmp_path, argv):
        """The CLI exits with an error and writes no file at all."""
        with pytest.raises(SystemExit) as excinfo:
            exact.main([*argv, "--output-dir", str(tmp_path)])
        assert excinfo.value.code != 0
        assert not list(tmp_path.glob("*.csv"))

    def test_default_run_writes_exactly_seven_files(self, exact, tmp_path):
        """The default scope is the six exact tables plus the fixed-action table."""
        assert exact.main(["--output-dir", str(tmp_path)]) == 0
        written = sorted(path.name for path in tmp_path.glob("*.csv"))
        assert written == sorted(
            [
                *(
                    f"optimal_policy_onemax_{n}_{algorithm}.csv"
                    for n in SIZES
                    for algorithm in ALGORITHMS
                ),
                "exact_fixed_action_counts.csv",
            ]
        )


class TestDeterminism:
    """The script has no random component: identical runs give identical files."""

    def test_identical_runs_are_byte_identical(self, exact, tmp_path):
        """Two runs into two directories produce byte-identical CSVs."""
        outputs = []
        for run in range(2):
            out_dir = tmp_path / f"run{run}"
            assert exact.main(["--output-dir", str(out_dir)]) == 0
            outputs.append(
                {path.name: path.read_bytes() for path in sorted(out_dir.glob("*.csv"))}
            )
        assert len(outputs[0]) == 7
        assert outputs[0] == outputs[1]

    def test_shipped_files_are_up_to_date(self, exact):
        """The checked-in tables are exactly what the script produces."""
        assert exact.main(["--check"]) == 0
        for n in SIZES:
            for algorithm in ALGORITHMS:
                path = policy_table_path(n, algorithm)
                text = exact.policy_table(n, algorithm, PORTFOLIO).to_csv(index=False)
                assert path.read_text() == text, path.name

    def test_check_detects_a_stale_file(self, exact, tmp_path):
        """``--check`` reports a stale or missing table and writes nothing."""
        assert exact.main(["--output-dir", str(tmp_path)]) == 0
        path = tmp_path / "optimal_policy_onemax_50_one_plus_one.csv"
        path.write_text("f_x,action,choice,mean_evals,n_seeds,ci_low,ci_high\n")
        assert exact.main(["--output-dir", str(tmp_path), "--check"]) == 1
        # --check never writes
        assert path.read_text().startswith("f_x,action,choice,mean_evals")
        path.unlink()
        assert exact.main(["--output-dir", str(tmp_path), "--check"]) == 1


class TestBaselineIntegration:
    """The exact tables are usable through the package's baseline API."""

    @pytest.mark.parametrize(
        ("n", "algorithm"), list(itertools.product(SIZES, ALGORITHMS))
    )
    def test_best_static_uses_the_exact_policy(self, n, algorithm):
        """``best_static`` selects the exact optimal action of the bucket."""
        policy = best_static(problem="OneMax", n=n, algorithm=algorithm)
        table = load_policy_table(policy_table_path(n, algorithm))
        expected = optimal_actions(table)
        env = make_env("OneMax", n, algorithm, PORTFOLIO)
        for start in (0, n // 2, n - 1):
            env.x = OneMax(n=n, rng=np.random.default_rng(0), initObj=start)
            assert policy(env, None) == expected[start]


class TestPortfolioScope:
    """The exact tables are optimal over the shipped portfolio, not globally.

    ``r = n`` flips every bit, so the (1+1) EA complements the string: from
    ``f(x) = 0`` that reaches the optimum in exactly one evaluation.  The shipped
    portfolio stops at 16 for comparability with the Theory benchmark, so the
    tables are portfolio-optimal; a policy that uses ``r = n`` lies outside their
    action set and is not comparable.
    """

    def test_full_complement_solves_the_hard_start_in_one_evaluation(self, exact):
        """``r = n`` solves ``f(x) = 0`` in one evaluation, outside the portfolio."""
        portfolio = [1, 2, 4, 8]
        full = exact.solve(8, "one_plus_one", portfolio)
        assert full["values"][0] == pytest.approx(1.0)
        # the policy stores an index into the portfolio
        assert portfolio[int(full["policy"][0])] == 8

        shipped = exact.solve(8, "one_plus_one", [1, 2, 4])
        assert shipped["values"][0] > full["values"][0] * 5

    def test_widening_lambda_does_not_beat_the_constant_optimum(self, exact):
        """For the (1+lambda) EA the portfolio is not a restriction at n = 20."""
        narrow = exact.solve(20, "one_plus_lambda", [1, 2, 4, 8, 16])
        wide = exact.solve(20, "one_plus_lambda", list(range(1, 21)))
        assert wide["values"][0] == pytest.approx(narrow["values"][0])


class TestCheckTolerance:
    """``--check`` ignores rounding noise but still reports a real change."""

    def test_tolerance_ignores_rounding_and_reports_real_changes(self, exact, tmp_path):
        """A 1e-10 relative perturbation passes; a 1 % perturbation is stale."""
        assert exact.main(["--output-dir", str(tmp_path), "--check"]) == 1  # empty
        assert exact.main(["--output-dir", str(tmp_path)]) == 0

        name = "optimal_policy_onemax_50_one_plus_one.csv"
        path = tmp_path / name
        original = path.read_text()

        noisy = pd.read_csv(path)
        noisy["mean_evals"] = noisy["mean_evals"] * (1.0 + 1e-10)
        path.write_text(noisy.to_csv(index=False))
        assert exact.main(["--output-dir", str(tmp_path), "--check"]) == 0

        shifted = pd.read_csv(path)
        shifted["mean_evals"] = shifted["mean_evals"] * 1.01
        path.write_text(shifted.to_csv(index=False))
        assert exact.main(["--output-dir", str(tmp_path), "--check"]) == 1

        # a changed action column is a real change, whatever the floats do
        path.write_text(original)
        reordered = pd.read_csv(path)
        reordered.loc[0, "action"] = 4
        path.write_text(reordered.to_csv(index=False))
        assert exact.main(["--output-dir", str(tmp_path), "--check"]) == 1
