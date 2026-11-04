"""Run the baselines of the GA benchmark and print a table.

Evaluates the three table-free baselines of
``dacbench/envs/env_utils/ga_baselines.py`` -- ``theory``, ``success_rule`` and
``random`` -- on the shipped OneMax and LeadingOne instance sets, for n = 50, 100,
200 and all three algorithms of the benchmark.

The benchmark ships no reference policy tables, so the reference is the **best
fixed action measured in the run** (Biedenkapp et al., GECCO 2022): every
portfolio value is played as a constant policy in its own fresh environment and
the cheapest one becomes ``best_fixed`` and the denominator of the ``ratio``
column. Because the winner is chosen on the same seeds it is read on, its own
ratio of 1.00 is optimistic (winner's curse). ``theory`` has no policy for the
(1+(lambda,lambda)) GA on LeadingOne, so that combination reports only
``best_fixed``, ``success_rule`` and ``random``.

The comparison is **paired**: every policy of a configuration -- including every
constant action behind ``best_fixed`` -- runs in its own fresh environment, so
episode ``k`` of each meets the same seed and therefore the same instance of the
round-robin ladder; the table shows the effect of the policy, not of the instance
draw. Episodes are driven directly (``env.reset`` / ``env.step``, no runner) in a
test environment with a budget of ``cutoff = 10 n^2`` evaluations.

The default instance sets are the *ladders* (``onemax_{n}.csv``,
``leadingones_{n}.csv``, several initial solutions each) rather than the
single-instance ``*_random`` files: the ladders cover the number of correct bits
in the initial solution, which is the difficulty axis of the benchmark. The paths
are absolute on purpose, because ``GABenchmark`` looks up a same-named CSV in the
current working directory first.

Run from the repository root::

    python examples/run_ga_baselines.py
    python examples/run_ga_baselines.py --quick
    python examples/run_ga_baselines.py --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from dacbench.benchmarks import GABenchmark
from dacbench.envs.env_utils.ga_baselines import (
    ALGORITHMS,
    PROBLEM_FILE_STEMS,
    random_policy,
    success_rule,
    theory_policy,
)

#: Problem name -> instance-set file stem.
INSTANCE_FILE_STEMS = PROBLEM_FILE_STEMS

#: Directory with the shipped instance sets (absolute, see the module docstring).
INSTANCE_DIR = Path(__file__).resolve().parents[1] / "dacbench" / "instance_sets" / "ga"

#: Portfolio of the environment; the constant actions of ``best_fixed`` are these.
ACTION_CHOICES = [1, 2, 4, 8, 16]

#: Evaluation budget factor of the test environment (cutoff = factor * n**2).
CUTOFF_FACTOR = 10

#: Confidence level of the reported interval (normal approximation).
Z_95 = 1.96

#: Adaptive policies of the comparison; ``best_fixed`` is measured separately.
POLICIES = ("theory", "success_rule", "random")


def make_env(problem, n, algorithm, action_choices=ACTION_CHOICES):
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


def constant_policy(index):
    """Return a policy that always plays the action ``index``."""

    def policy(env, state):
        # API: none
        return int(index)

    policy.reset = lambda: None
    return policy


def make_policy(name, problem, algorithm, seed):
    """Create a fresh policy instance (state must not leak across episodes)."""
    if name == "theory":
        return theory_policy(problem, algorithm)
    if name == "success_rule":
        return success_rule(algorithm=algorithm, initial_value=1, factor=1.5)
    if name == "random":
        return random_policy(seed=seed)
    raise ValueError(f"Error: unknown policy {name!r}; expected one of {POLICIES}")


def confidence_interval(values):
    """Return the half-width of the 95% CI of the mean (normal approximation)."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return float("nan")
    return float(Z_95 * values.std(ddof=1) / np.sqrt(values.size))


def run_policy(problem, n, algorithm, make, seeds):
    """Run one policy over ``seeds``; return ``(evaluations, solved)``.

    ``make`` is called per episode, so stateful policies start from scratch. The
    environment is created here, once per policy, so that a configuration's
    policies all meet the same (seed, instance) pairs.
    """
    env = make_env(problem, n, algorithm)
    evals, solved = [], []
    for seed in seeds:
        policy = make(seed)
        policy.reset()
        state, _ = env.reset(seed=seed)
        while True:
            action = policy(env, state)
            state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        evals.append(int(env.total_evals))
        solved.append(bool(env.x.is_optimal()))
    return evals, solved


def make_row(problem, n, algorithm, name, evals, solved, reference):
    """Assemble one output row."""
    mean = float(np.mean(evals))
    return {
        "problem": problem,
        "n": n,
        "algorithm": algorithm,
        "policy": name,
        "mean_evals": mean,
        "ci": confidence_interval(evals),
        "ratio": None if reference is None else mean / reference,
        "solved": f"{sum(solved)}/{len(solved)}",
    }


def evaluate(problems, sizes, algorithms, seeds):
    """Evaluate all policies and return (result rows, combinations without theory).

    Every policy of a configuration runs in its own fresh environment and meets
    the seeds in the same order, so all of them see the same (seed, instance)
    pairs.
    """
    rows, unavailable = [], []
    for problem in problems:
        for n in sizes:
            for algorithm in algorithms:
                fixed = [
                    run_policy(
                        problem,
                        n,
                        algorithm,
                        lambda _seed, index=index: constant_policy(index),
                        seeds,
                    )
                    for index in range(len(ACTION_CHOICES))
                ]
                best = min(range(len(fixed)), key=lambda i: np.mean(fixed[i][0]))
                reference = float(np.mean(fixed[best][0]))
                rows.append(
                    make_row(
                        problem,
                        n,
                        algorithm,
                        f"best_fixed(r={ACTION_CHOICES[best]})",
                        *fixed[best],
                        reference,
                    )
                )
                for name in POLICIES:
                    try:
                        make_policy(name, problem, algorithm, 0)
                    except ValueError as error:
                        unavailable.append((problem, n, algorithm, str(error)))
                        continue
                    rows.append(
                        make_row(
                            problem,
                            n,
                            algorithm,
                            name,
                            *run_policy(
                                problem,
                                n,
                                algorithm,
                                lambda seed, nm=name, p=problem, a=algorithm: (
                                    make_policy(nm, p, a, seed)
                                ),
                                seeds,
                            ),
                            reference,
                        )
                    )
    return rows, unavailable


def print_table(rows):
    """Print the result table."""
    header = (
        f"{'problem':<11}{'n':>4}  {'algorithm':<25}{'policy':<20}"
        f"{'mean evals':>11}  {'95% CI':>9}  {'ratio':>6}  {'solved':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        ci = "-" if np.isnan(row["ci"]) else f"{row['ci']:.1f}"
        ratio = "-" if row["ratio"] is None else f"{row['ratio']:.2f}"
        print(
            f"{row['problem']:<11}{row['n']:>4}  {row['algorithm']:<25}"
            f"{row['policy']:<20}{row['mean_evals']:>11.1f}  "
            f"{ci:>9}  {ratio:>6}  {row['solved']:>7}"
        )


def summary(problems, seeds, unavailable):
    """Return the footer lines printed below the table."""
    stems = ", ".join(sorted(INSTANCE_FILE_STEMS[problem] for problem in problems))
    lines = [
        f"\n{len(seeds)} episode(s) per policy and configuration, evaluation budget "
        f"cutoff = {CUTOFF_FACTOR} n^2, test mode, instance sets: {stems}. "
        f"Every policy -- and every constant action behind 'best_fixed' -- ran in "
        f"a fresh environment, so all of them saw the same (seed, instance) pairs. "
        f"'ratio' is the mean evaluations of a policy relative to the best fixed "
        f"action measured on these same seeds, which makes that action's own 1.00 "
        f"optimistic. '95% CI' is a normal approximation, shown as '-' for a "
        f"single episode; a single episode is one sample of a random search, so "
        f"the table is indicative and not a measurement of the expected runtime."
    ]
    if unavailable:
        combos = ", ".join(
            f"{problem} n={n} {algorithm}" for problem, n, algorithm, _ in unavailable
        )
        lines.append(
            f"\nNo theory policy for {len(unavailable)} configuration(s): {combos}. "
            f"Those combinations still show the best fixed action, the success "
            f"rule and the random policy; the reason is printed below."
        )
        lines.append(f"  {unavailable[0][3]}")
    return lines


def main(argv=None):
    """Parse arguments and run the evaluation."""
    parser = argparse.ArgumentParser(
        description="Run the baselines of the GA benchmark.",
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

    rows, unavailable = evaluate(args.problems, sizes, args.algorithms, seeds)
    print_table(rows)
    for line in summary(args.problems, seeds, unavailable):
        print(line)
    return rows


if __name__ == "__main__":
    main()
