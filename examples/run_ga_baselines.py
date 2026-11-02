"""Run baseline policies for the GA benchmark and print a comparison table.

The script evaluates three policies on the shipped OneMax and LeadingOne instance
sets, for n = 50, 100, 200 and all three algorithms of the benchmark:

- ``best_static``: the per-bucket greedy policy of the reference tables
  (``dacbench/instance_sets/ga/optimal_policy_*.csv``) -- exact for OneMax with
  the (1+1) EA and the (1+λ) EA, simulation-derived for the other four,
- ``success_rule``: the one-fifth success rule,
- ``random``: uniform random actions.

``best_static`` raises a ``ValueError`` for the eight combinations that have no
validated reference table (OneMax n=50 with the (1+(λ,λ)) GA, LeadingOne n=50
with all three algorithms, LeadingOne n=100 with the (1+λ) EA and LeadingOne
n=200 with all three algorithms); the script skips it there and their reference
is ``success_rule`` or the measured best fixed action.  Every combination remains
usable for training and evaluation.

Every episode is driven directly (``env.reset`` / ``env.step``) in a plain loop
with ``test_env=True``; no logger or runner machinery is involved. The evaluation
budget of the test environment is the ``cutoff`` passed below.

The comparison is **paired**: every policy gets its own fresh environment, so
episode ``k`` of every policy starts from the same seed and hence from the same
instance of the ladder. The printed ``ratio`` column divides a policy's mean
number of evaluations by that of ``best_static`` (where it is available), so the
table shows the effect of the policy and not of the instance draw. The ``gap``
column is the mean per-episode optimality gap ``evals / optimal_evals`` from
``ga_baselines.gap``: it exists only for the six exact OneMax combinations and
is computed against the exact optimum of each episode's start bucket; censored
(cutoff) episodes are left out, and every other combination prints ``n/a``
because no exact optimum exists there.

The default instance sets are the *ladders* (``onemax_{n}.csv``,
``leadingones_{n}.csv``, several initial solutions each), not the single-instance
``*_random`` files: the ladders cover the difficulty axis of the benchmark (the
number of correct bits in the initial solution), while a single random start
would make the table depend on one draw. The environment progresses through the
ladder round-robin, so consecutive seeds see different initial solutions and the
reported mean is an average over seeds *and* initial solutions.

The instance-set paths are absolute: ``GABenchmark`` looks up a same-named CSV in
the current working directory first, so a relative name would silently change
which instances are evaluated.

Run from the repository root::

    python examples/run_ga_baselines.py
    python examples/run_ga_baselines.py --quick
    python examples/run_ga_baselines.py --seeds 0 1 2 3 4

The derived tables are produced by
``dacbench/instance_sets/ga/derive_optimal_policy.py``; if they are missing, the
script still runs the other two policies and says so.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from dacbench.benchmarks import GABenchmark
from dacbench.envs.env_utils.ga_baselines import (
    ALGORITHMS,
    EXACT_COMBINATIONS,
    UNVALIDATED_COMBINATIONS,
    best_static,
    best_static_portfolio,
    gap,
    is_validated,
    random_policy,
    success_rule,
)

#: Problem name -> instance-set file stem.
INSTANCE_FILE_STEMS = {"OneMax": "onemax", "LeadingOne": "leadingones"}

#: Directory with the shipped instance sets (absolute, see the module docstring).
INSTANCE_DIR = Path(__file__).resolve().parents[1] / "dacbench" / "instance_sets" / "ga"

#: Fallback portfolio if no derived table (and hence no portfolio) is available.
DEFAULT_ACTION_CHOICES = [1, 2, 4, 8, 16]

#: Evaluation budget factor of the test environment (cutoff = factor * n**2).
CUTOFF_FACTOR = 10

#: Confidence level of the reported interval (normal approximation).
Z_95 = 1.96


def make_env(problem, n, algorithm, action_choices):
    """Create a test-mode GA environment for one (problem, n, algorithm)."""
    config = {
        "problem": problem,
        "algorithm": algorithm,
        "action_choices": list(action_choices),
        "instance_set_path": str(
            INSTANCE_DIR / f"{INSTANCE_FILE_STEMS[problem]}_{n}.csv"
        ),
        "cutoff": CUTOFF_FACTOR * n * n,
    }
    return GABenchmark(config=config).get_environment(test_env=True)


def run_episode(env, policy, seed):
    """Run one episode with ``policy``; return ``(evaluations, solved)``."""
    policy.reset()
    state, _ = env.reset(seed=seed)
    start_f = int(env.x.fitness)
    while True:
        action = policy(env, state)
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    # ``init_obj`` is the fitness after reset, i.e. the start bucket of the
    # episode; it is needed by the optimality gap, which is per start bucket
    assert int(env.init_obj) == start_f
    return int(env.total_evals), bool(env.x.is_optimal())


def confidence_interval(values):
    """Return the half-width of the 95% CI of the mean (normal approximation)."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return float("nan")
    return float(Z_95 * values.std(ddof=1) / np.sqrt(values.size))


def make_policy(name, problem, n, algorithm, seed):
    """Create a fresh policy instance (state must not leak across episodes)."""
    if name == "best_static":
        return best_static(problem=problem, n=n, algorithm=algorithm)
    if name == "success_rule":
        return success_rule(algorithm=algorithm, initial_value=1, factor=1.5)
    return random_policy(seed=seed)


def combo_action_choices(problem, n, algorithm):
    """Return the portfolio of a combination and why ``best_static`` may be skipped.

    Returns:
    -------
    list of int
        portfolio the environment is configured with
    str | None
        reason to skip ``best_static``, ``None`` if it is available

    """
    if not is_validated(problem, n, algorithm):
        entry = UNVALIDATED_COMBINATIONS[(INSTANCE_FILE_STEMS[problem], n, algorithm)]
        return DEFAULT_ACTION_CHOICES, f"no validated table ({entry['category']})"
    try:
        return best_static_portfolio(problem, n, algorithm), None
    except FileNotFoundError:
        return DEFAULT_ACTION_CHOICES, "table missing (run the derive script)"


def optimality_gap(problem, n, algorithm, evaluations, solved, starts):
    """Return the mean per-episode optimality gap, or ``None`` if unavailable.

    The gap of an episode is ``evals / optimal_evals`` from
    :func:`dacbench.envs.env_utils.ga_baselines.gap`, computed against the
    exact optimum of *that* episode's start bucket. Censored episodes (cutoff
    reached) have no ratio and are left out; if every episode was censored, the
    mean is undefined and ``None`` is returned.

    Returns:
    -------
    float | None
        Mean optimality gap over the solved episodes, ``None`` for a
        non-exact combination (no exact reference exists).

    """
    if (INSTANCE_FILE_STEMS[problem], n, algorithm) not in EXACT_COMBINATIONS:
        return None
    ratios = []
    for episode_evals, episode_solved, start_f in zip(
        evaluations, solved, starts, strict=True
    ):
        result = gap(
            # GAEnv.total_evals counts the initial individual, the exact
            # reference V[start_f] does not
            episode_evals - 1,
            problem,
            n,
            algorithm,
            start_f,
            solved=episode_solved,
        )
        if result["ratio"] is not None:
            ratios.append(result["ratio"])
    return float(np.mean(ratios)) if ratios else None


def evaluate(problems, sizes, algorithms, seeds, policy_names):
    """Evaluate all policies and return (result rows, skipped combinations).

    Every policy of a combination runs in its own fresh environment and meets the
    seeds in the same order, so all policies see the same (seed, instance) pairs.
    """
    rows = []
    skipped = []
    for problem in problems:
        for n in sizes:
            for algorithm in algorithms:
                action_choices, skip_reason = combo_action_choices(
                    problem, n, algorithm
                )
                names = list(policy_names)
                if skip_reason is not None:
                    skipped.append((problem, n, algorithm, skip_reason))
                    names = [name for name in names if name != "best_static"]
                combo_rows = []
                for name in names:
                    # one fresh environment per policy: the ladder index starts
                    # at 0 again, so episode k of every policy uses the same
                    # instance
                    env = make_env(problem, n, algorithm, action_choices)
                    evals, solved = [], []
                    starts = []
                    for seed in seeds:
                        policy = make_policy(name, problem, n, algorithm, seed)
                        n_evals, ok = run_episode(env, policy, seed)
                        evals.append(n_evals)
                        solved.append(ok)
                        starts.append(int(env.init_obj))
                    combo_rows.append(
                        {
                            "problem": problem,
                            "n": n,
                            "algorithm": algorithm,
                            "policy": name,
                            "mean_evals": float(np.mean(evals)),
                            "ci": confidence_interval(evals),
                            "gap": optimality_gap(
                                problem, n, algorithm, evals, solved, starts
                            ),
                            "solved": f"{sum(solved)}/{len(solved)}",
                        }
                    )
                reference = next(
                    (
                        row["mean_evals"]
                        for row in combo_rows
                        if row["policy"] == "best_static"
                    ),
                    None,
                )
                for row in combo_rows:
                    row["ratio"] = (
                        float("nan")
                        if reference is None
                        else row["mean_evals"] / reference
                    )
                rows.extend(combo_rows)
    return rows, skipped


def print_table(rows):
    """Print the result table."""
    header = (
        f"{'problem':<11}{'n':>4}  {'algorithm':<25}{'policy':<13}"
        f"{'mean evals':>11}  {'95% CI':>9}  {'ratio':>6}  {'gap':>7}  {'solved':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        ratio = "     -" if not np.isfinite(row["ratio"]) else f"{row['ratio']:6.2f}"
        gap_text = "    n/a" if row["gap"] is None else f"{row['gap']:7.3f}"
        print(
            f"{row['problem']:<11}{row['n']:>4}  {row['algorithm']:<25}"
            f"{row['policy']:<13}{row['mean_evals']:>11.1f}  "
            f"{row['ci']:>9.1f}  {ratio}  {gap_text}  {row['solved']:>7}"
        )


def main(argv=None):
    """Parse arguments and run the evaluation."""
    parser = argparse.ArgumentParser(
        description="Run baseline policies for the GA benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="seeds of the evaluated episodes",
    )
    parser.add_argument(
        "--n",
        nargs="+",
        type=int,
        default=[50, 100, 200],
        help="problem sizes to evaluate",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        choices=list(INSTANCE_FILE_STEMS),
        default=list(INSTANCE_FILE_STEMS),
        help="problems to evaluate",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=list(ALGORITHMS),
        default=list(ALGORITHMS),
        help="algorithms to evaluate",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="small configuration for a quick check (n=50, one seed, both problems)",
    )
    args = parser.parse_args(argv)

    sizes = [50] if args.quick else args.n
    seeds = [0] if args.quick else args.seeds
    policy_names = ["best_static", "success_rule", "random"]

    rows, skipped = evaluate(args.problems, sizes, args.algorithms, seeds, policy_names)
    print_table(rows)
    print(
        f"\n{len(seeds)} episode(s) per configuration, evaluation budget "
        f"cutoff = {CUTOFF_FACTOR} n^2, test mode, instance sets "
        f"{sorted(INSTANCE_FILE_STEMS.values())}. Every policy ran in a fresh "
        f"environment, so all policies saw the same (seed, instance) pairs; "
        f"``ratio`` is relative to ``best_static``. ``gap`` is the mean "
        f"``evals / optimal_evals`` over the solved episodes for the six exact "
        f"OneMax combinations (exact DP, per start bucket) and ``n/a`` "
        f"elsewhere: the other combinations have no exact optimum, only a "
        f"best-known reference. A single episode can score below 1 because the "
        f"reference is an expectation - for these six combinations "
        f"``best_static`` *is* the exact optimal policy, so its gap approaches "
        f"1 as the number of seeds grows."
    )
    if skipped:
        print(
            f"\nbest_static skipped for {len(skipped)} configuration(s): "
            + ", ".join(
                f"{problem} n={n} {algorithm} ({reason})"
                for problem, n, algorithm, reason in skipped
            )
            + ". ``best_static`` raises a ValueError for the eight combinations "
            "without a validated table; their reference is `success_rule` or the "
            "measured best fixed action, see docs/source/benchmark_docs/ga.rst."
        )
    return rows


if __name__ == "__main__":
    main()
