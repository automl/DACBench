"""Baseline and reference policies for the GA benchmark.

The GA benchmark (:mod:`dacbench.envs.ga`) controls the mutation strength ``r`` of
the (1+1) EA or the offspring population size ``lambda`` of the (1+lambda) EA and
the (1+(lambda,lambda)) GA. This module provides three policies that are useful as
reference points for DAC methods:

``best_static``
    Looks up the current fitness bucket ``f(x)`` in a derived reference table
    and returns the action with the smallest expected number of evaluations in
    that bucket. Tables ship for 9 of the 18 (problem, n, algorithm)
    combinations; for the other nine it raises a ``ValueError`` because no
    validated table exists (see below).
``success_rule``
    The classic one-fifth success rule, applied either to ``r`` (``one_plus_one``)
    or to ``lambda`` (the population-based algorithms); for the (1+lambda) EA
    the lambda-update is an *adaptation* without a published analysis.
``random_policy``
    Uniform random action, the trivial baseline.

Policy API
----------
Every policy is a callable ``policy(env, state) -> int`` returning an action
**index** in ``[0, len(env.action_choices))``, matching
:class:`dacbench.agents.generic_agent.GenericAgent`, which calls
``policy(env, state)``. ``state`` is the observation returned by the environment
(a numpy array here) and is not used by these policies; the parameter is kept for
API compatibility. Policies that keep state expose ``policy.reset()``.

Derived reference tables (CSV contract)
---------------------------------------
``best_static`` reads one file per (problem, n, algorithm) combination:

``dacbench/instance_sets/ga/optimal_policy_{problem}_{n}_{algorithm}.csv``

with ``problem`` in ``{onemax, leadingones}``, ``n`` in ``{50, 100, 200}`` and
``algorithm`` in ``{one_plus_one, one_plus_lambda, one_plus_lambda_lambda}``.
A table ships for 9 of these 18 combinations: OneMax n=50 with the (1+1) EA and
the (1+lambda) EA, OneMax n=100 with the (1+1) EA and the (1+(lambda,lambda))
GA, all three OneMax n=200 combinations, LeadingOne n=100 with the (1+1) EA and
the (1+(lambda,lambda)) GA. A table ships only if the derived policy needs at
most 1.05 times the evaluations of the best fixed action on **both** paired
seed sets of the validation (A: seeds 0-11, the selection protocol; B: the
held-out seeds 1000-1011) and is never significantly worse. The other nine have
no validated table, because there the derived policy was not demonstrably better
than the best fixed action (see :data:`UNVALIDATED_COMBINATIONS`);
:func:`best_static` raises a ``ValueError`` for them, and
``derive_optimal_policy.py --all`` can still generate them for research. The
columns are, in this order::

    f_x,action,choice,mean_evals,n_seeds,ci_low,ci_high

One row per (fitness bucket ``f_x``, portfolio index ``action``), sorted by
``f_x`` and then ``action``. ``choice`` is the parameter value (``r`` or
``lambda``) of that portfolio entry and ``mean_evals`` is the estimated number
of evaluations needed to reach the optimum when running the bucket's fixed
action from fitness ``f_x``. ``mean_evals`` may be the literal ``inf``: no
simulated sample of that action improved in that bucket, so the improvement
probability of the sample is zero and no finite estimate exists. ``inf`` is a
statement about the simulated sample, not about the algorithm, and such rows are
never selected.

The buckets are **simulation-derived**. For every fitness bucket ``f(x)`` and
every portfolio action, one generation is simulated many times with common
random numbers (the samples of all actions of a bucket are paired), which
estimates the expected evaluations per improvement and the distribution of the
successor bucket. A backward dynamic programme from ``f(x) = n`` (cost 0) then
computes the expected cost to the optimum,
``V(b, a) = E[e] / q + E[V(b') | b' > b]`` with ``q = P(b' > b)``, if action
``a`` is played in bucket ``b`` and the derived policy afterwards. The CSV
column ``mean_evals`` holds this ``V(b, a)``. ``best_static`` returns the
per-bucket ``argmin_a V(b, a)``, i.e. exactly the action the derived dynamic
programme selects. This follows the methodology of Biedenkapp, Dang, Krejca,
Hutter & Doerr, "Theory-inspired parameter control benchmarks for dynamic
algorithm configuration", GECCO 2022, arXiv:2202.03259,
DOI 10.1145/3512290.3528846, who derive optimal portfolio-based control policies
for the mutation rate of RLS on LeadingOnes.

Unlike that work, the tables for the GA benchmark are *not* backed by a
theorem: they depend on the fitness-bucket abstraction (the state is only
``f(x)``, so bit-level correlations from the history are ignored), and the
estimates carry Monte-Carlo error (``n_seeds``, ``ci_low``, ``ci_high``). The
derived policy is optimal *within* this simulation model, not for the true
process: these are **approximate reference policies, not optima**, and their
measured advantage over the best fixed action is modest (0.79-1.01 times its
evaluations where a table ships; OneMax n=100 with the (1+1) EA is at parity on
the held-out protocol B, 1.007). A row with ``mean_evals = inf`` means that no
simulated sample of that action improved (``q = 0`` in the sample), not that the
action can never improve.

Prior art for the baseline rules
--------------------------------
- One-fifth success rule for a step size (here: the mutation strength ``r``):
  H.-P. Schwefel / I. Rechenberg (1973), reviewed and cast into the
  multiplicative form used here by S. Kern, S. D. Mueller, N. Hansen,
  D. Bueche, J. Ocenasek & P. Koumoutsakos, "Learning probability distributions
  in continuous evolutionary algorithms - a comparative review", Natural
  Computing 3:77-112, 2004, DOI 10.1023/B:NACO.0000023416.59689.4e (Eq. 3:
  multiply by ``F`` on success, by ``F**(-1/4)`` on failure, with ``F = 1.5``);
  see also A. Auger, "Benchmarking the (1+1) evolution strategy with one-fifth
  success rule on the BBOB-2009 function testbed", GECCO 2009 (Table 1).
- One-fifth success rule for the population size ``lambda`` of the
  (1+(lambda,lambda)) GA: B. Doerr & C. Doerr, "Optimal parameter choices
  through self-adjustment: applying the 1/5-th rule in discrete settings",
  GECCO 2015, arXiv:1504.03212 (success: ``lambda <- lambda/F``;
  failure: ``lambda <- lambda * F**(1/4)``), and B. Doerr & C. Doerr,
  "Optimal static and self-adjusting parameter choices for the
  (1+(lambda,lambda)) genetic algorithm", Algorithmica 80:1752-1801, 2018,
  DOI 10.1007/s00453-017-0354-9.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: Maps the benchmark's problem names to the file-name stem of the derived tables.
PROBLEM_FILE_STEMS = {"OneMax": "onemax", "LeadingOne": "leadingones"}

#: Algorithms for which derived tables and success rules exist.
ALGORITHMS = ("one_plus_one", "one_plus_lambda", "one_plus_lambda_lambda")

#: Directory with the shipped derived tables.
DEFAULT_INSTANCE_DIR = Path(__file__).resolve().parents[2] / "instance_sets" / "ga"

#: Columns ``best_static`` needs from a derived table (the rest is metadata).
_REQUIRED_COLUMNS = ("f_x", "action", "choice", "mean_evals")

#: Category of a combination without a validated table: the derivation ran, but
#: the derived policy failed the full-episode validation (it was not at most
#: 1.05 times the best fixed action on both paired seed sets).
FAILED_VALIDATION = "failed validation"
#: Category of a combination without a validated table: the method itself is
#: structurally unsuited (LeadingOne), so a validation run would not change the
#: verdict.
STRUCTURAL = "structural"

#: Why the LeadingOne combinations are structurally unvalidated.
UNVALIDATED_REASON = (
    "the fitness-only bucket abstraction is insufficient for LeadingOne: the "
    "suffix after the leading ones drives mutation and crossover, but it is "
    "not part of f(x) and therefore not part of the derived policy"
)

#: Portfolio of the shipped derived tables (``choice`` column).
_PORTFOLIO = (1, 2, 4, 8, 16)

#: The nine (problem, n, algorithm) combinations whose derived table ships: the
#: ones whose policy was at most 1.05 times the best fixed action on *both*
#: paired seed sets (A and B) and never significantly worse.
SHIPPED_COMBINATIONS = (
    ("onemax", 50, "one_plus_one"),
    ("onemax", 50, "one_plus_lambda"),
    ("onemax", 100, "one_plus_one"),
    ("onemax", 100, "one_plus_lambda_lambda"),
    ("onemax", 200, "one_plus_one"),
    ("onemax", 200, "one_plus_lambda"),
    ("onemax", 200, "one_plus_lambda_lambda"),
    ("leadingones", 100, "one_plus_one"),
    ("leadingones", 100, "one_plus_lambda_lambda"),
)

#: The nine combinations without a validated table. Every entry gives the
#: category, the measured ratios of the derived policy's evaluations to those of
#: the best fixed action (``A``: seeds 0-11, the selection protocol; ``B``: the
#: held-out seeds 1000-1011; ``None`` where the ratio was not measured), the best
#: fixed portfolio value and a one-line reason. Both protocols run a fresh
#: environment per policy, so every policy sees the same (seed, instance) pairs.
UNVALIDATED_COMBINATIONS = {
    ("onemax", 50, "one_plus_lambda_lambda"): {
        "category": FAILED_VALIDATION,
        "ratios": {"A": 1.047, "B": 1.088},
        "best_choice": 1,
        "reason": "worse than fixed lambda = 1 on both seed sets (already on the "
        "selection protocol A, and above the 1.05 tolerance on held-out B)",
    },
    ("onemax", 100, "one_plus_lambda"): {
        "category": FAILED_VALIDATION,
        "ratios": {"A": 1.041, "B": 1.088},
        "best_choice": 1,
        "reason": "worse than fixed lambda = 1 on both seed sets (already on the "
        "selection protocol A, and above the 1.05 tolerance on held-out B)",
    },
    ("leadingones", 50, "one_plus_one"): {
        "category": STRUCTURAL,
        "ratios": {"A": 1.095, "B": None},
        "best_choice": 1,
        "reason": "worse than fixed r = 1 on the selection seeds",
    },
    ("leadingones", 50, "one_plus_lambda"): {
        "category": STRUCTURAL,
        "ratios": {"A": 1.136, "B": None},
        "best_choice": 2,
        "reason": "worse than fixed lambda = 2 on the selection seeds",
    },
    ("leadingones", 50, "one_plus_lambda_lambda"): {
        "category": STRUCTURAL,
        "ratios": {"A": 1.083, "B": None},
        "best_choice": 1,
        "reason": "worse than fixed lambda = 1 on the selection seeds",
    },
    ("leadingones", 100, "one_plus_lambda"): {
        "category": FAILED_VALIDATION,
        "ratios": {"A": 0.953, "B": 1.357},
        "best_choice": 2,
        "reason": "fine on the selection protocol A but clearly worse than "
        "fixed lambda = 2 on held-out B",
    },
    ("leadingones", 200, "one_plus_one"): {
        "category": FAILED_VALIDATION,
        "ratios": {"A": 1.016, "B": 1.062},
        "best_choice": 1,
        "reason": "worse than fixed r = 1 on both seed sets, and above the 1.05 "
        "tolerance on held-out B",
    },
    ("leadingones", 200, "one_plus_lambda"): {
        "category": STRUCTURAL,
        "ratios": {"A": 1.073, "B": None},
        "best_choice": 1,
        "reason": "worse than fixed lambda = 1 on the selection seeds",
    },
    ("leadingones", 200, "one_plus_lambda_lambda"): {
        "category": STRUCTURAL,
        "ratios": {"A": 1.269, "B": 1.316},
        "best_choice": 1,
        "reason": "worse than fixed lambda = 1 on both seed sets; the paired "
        "95% CI of the difference excludes 0 (the clearest failure)",
    },
}


def format_ratios(ratios):
    """Return the measured ratios of an unvalidated combination as a string."""
    return " / ".join(
        f"{label} {value:.3f}" for label, value in ratios.items() if value is not None
    )


def is_validated(problem, n, algorithm):
    """Return whether a validated reference table ships for a combination.

    Parameters
    ----------
    problem : str
        Problem name (``OneMax`` / ``LeadingOne``) or file-name stem.
    n : int
        Problem size.
    algorithm : str
        One of :data:`ALGORITHMS`.

    Returns:
    -------
    bool
        ``True`` if ``best_static`` can load a table for the combination,
        ``False`` if the combination is one of
        :data:`UNVALIDATED_COMBINATIONS`.

    """
    return (
        _problem_stem(problem),
        int(n),
        algorithm,
    ) not in UNVALIDATED_COMBINATIONS


def _unvalidated_error(problem_stem, n, algorithm):
    """Return the ``ValueError`` raised for a combination without a table."""
    entry = UNVALIDATED_COMBINATIONS[(problem_stem, int(n), algorithm)]
    best_choice = entry["best_choice"]
    if entry["category"] == STRUCTURAL:
        cause = UNVALIDATED_REASON
    else:
        cause = (
            "the derived policy is not at most 1.05 times the best fixed "
            "action on both paired seed sets, so it is not a useful "
            "reference"
        )
    return ValueError(
        f"Error: no validated reference table exists for problem "
        f"{problem_stem!r}, n={int(n)} and algorithm {algorithm!r}: {cause}. "
        f"Measured evaluations of the derived policy relative to the best "
        f"fixed action: {format_ratios(entry['ratios'])} (A: seeds 0-11, "
        f"B: held-out seeds 1000-1011); {entry['reason']}. Use "
        f"success_rule(algorithm={algorithm!r}) or the measured best fixed "
        f"action instead (portfolio value {best_choice}, action index "
        f"{_PORTFOLIO.index(best_choice)}; portfolio {list(_PORTFOLIO)}). See "
        f"docs/source/benchmark_docs/ga.rst for the measurements. To use an "
        f"unvalidated table anyway, generate it with derive_optimal_policy.py "
        f"--all and pass policy_path explicitly."
    )


def _problem_stem(problem):
    """Return the file-name stem for a problem name (''OneMax'' -> ''onemax'')."""
    if problem in PROBLEM_FILE_STEMS:
        return PROBLEM_FILE_STEMS[problem]
    if problem in PROBLEM_FILE_STEMS.values():
        return problem
    raise ValueError(
        f"Error: unknown problem {problem!r}; expected one of "
        f"{list(PROBLEM_FILE_STEMS)} (or their lower-case file stems)"
    )


def policy_file_name(problem, n, algorithm):
    """Return the file name of the derived table for one combination."""
    if algorithm not in ALGORITHMS:
        raise ValueError(
            f"Error: unknown algorithm {algorithm!r}; expected one of {ALGORITHMS}"
        )
    return f"optimal_policy_{_problem_stem(problem)}_{int(n)}_{algorithm}.csv"


def _policy_path(problem, n, algorithm, policy_path, instance_dir):
    """Return the path of the derived table for one (problem, n, algorithm).

    Raises:
    ------
    ValueError
        If the combination has no validated table (see
        :data:`UNVALIDATED_COMBINATIONS`) and no explicit path is given.

    """
    if policy_path is not None:
        return Path(policy_path)
    if not is_validated(problem, n, algorithm):
        raise _unvalidated_error(_problem_stem(problem), n, algorithm)
    directory = Path(instance_dir) if instance_dir is not None else DEFAULT_INSTANCE_DIR
    return directory / policy_file_name(problem, n, algorithm)


def load_policy_table(path):
    """Read and validate a derived reference table.

    ``mean_evals`` is coerced to float; the literal ``inf`` (and any value that
    cannot be parsed as a number) marks an action for which no simulated sample
    improved, and such a row is therefore never selected. The returned frame is
    sorted by ``f_x`` and ``action``.

    Raises:
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If a required column is missing.

    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Error: derived policy table not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Error: {path} is missing the column(s) {missing}; expected at least "
            f"{list(_REQUIRED_COLUMNS)}"
        )
    df = df.copy()
    df["f_x"] = pd.to_numeric(df["f_x"], errors="raise").astype(float)
    df["action"] = pd.to_numeric(df["action"], errors="raise").astype(int)
    df["choice"] = pd.to_numeric(df["choice"], errors="raise").astype(float)
    # "inf" parses directly; anything unparsable becomes NaN -> treated as inf
    df["mean_evals"] = pd.to_numeric(df["mean_evals"], errors="coerce").astype(float)
    return df.sort_values(["f_x", "action"]).reset_index(drop=True)


def _build_lookup(df):
    """Turn a table into ``{f_x: (actions, choices, mean_evals)}``."""
    lookup = {}
    for f_x, group in df.groupby("f_x", sort=True):
        group = group.sort_values("action")
        lookup[float(f_x)] = (
            group["action"].to_numpy(dtype=int),
            group["choice"].to_numpy(dtype=float),
            group["mean_evals"].to_numpy(dtype=float),
        )
    buckets = np.asarray(sorted(lookup), dtype=float)
    return lookup, buckets


def _best_row(lookup, buckets, f_x):
    """Return (action, choice, bucket) with the smallest finite ``mean_evals``.

    Buckets are considered in order of increasing distance to ``f_x`` (ties are
    resolved towards the smaller bucket); the first bucket that contains at
    least one finite ``mean_evals`` is used. Within that bucket the row with the
    smallest ``mean_evals`` wins, ties are resolved towards the smaller action
    index.

    Raises:
    ------
    ValueError
        If no bucket has a finite ``mean_evals`` at all.

    """
    order = np.argsort(np.abs(buckets - float(f_x)), kind="stable")
    for pos in order:
        bucket = float(buckets[pos])
        actions, choices, means = lookup[bucket]
        finite = np.isfinite(means)
        if not finite.any():
            continue
        masked = np.where(finite, means, np.inf)
        row = int(np.argmin(masked))
        return int(actions[row]), float(choices[row]), bucket
    raise ValueError(
        "Error: the derived policy table contains no finite mean_evals value; "
        "cannot select an action"
    )


def _map_choice_to_index(env, choice):
    """Map a portfolio value to the matching action index of ``env``.

    If the value is in ``env.action_choices`` its index is returned; otherwise
    the nearest value in log space is used (portfolio values are powers of two,
    so log space is the natural metric). Environments without an
    ``action_choices`` list (continuous action space, only ``one_plus_one``)
    receive the raw value truncated to an integer.
    """
    action_choices = getattr(env, "action_choices", None)
    if action_choices is None:
        # continuous action space (only ``one_plus_one``): the table values are
        # integers, so the raw value is the action
        return round(float(choice))
    choices = np.asarray(action_choices, dtype=float)
    match = np.flatnonzero(np.isclose(choices, choice))
    if match.size:
        return int(match[0])
    with np.errstate(divide="ignore"):
        log_choices = np.log(choices)
    index = int(np.argmin(np.abs(log_choices - np.log(choice))))
    return int(np.clip(index, 0, len(choices) - 1))


def best_static(
    problem=None, n=None, algorithm=None, policy_path=None, instance_dir=None
):
    """Return the per-bucket greedy policy of a derived reference table.

    The policy looks up the current fitness ``f(x)`` in the table identified by
    (problem, n, algorithm) and returns the action with the smallest finite
    ``mean_evals`` (the simulated cost-to-go ``V(f(x), a)``) for that bucket,
    i.e. the action of the derived dynamic programme. Buckets that are not in
    the table (gaps, including ``f(x)`` values above the largest bucket and the
    optimum) are resolved to the nearest bucket; ``inf`` rows are never
    selected.

    Parameters
    ----------
    problem, n, algorithm : optional
        Combination to load. Missing values are inferred from the environment at
        the first call: ``n`` from ``env.n``, ``algorithm`` from
        ``env.algorithm`` and the problem from ``env.problem.__name__`` (the
        environment stores the problem *class* ``OneMax`` / ``LeadingOne``, not
        its name). ``n`` only exists after ``reset()``, so an inferred ``n``
        requires the policy to be called in an episode.
    policy_path : str | Path | None
        Explicit table path instead of the naming scheme.
    instance_dir : str | Path | None
        Directory with the tables; defaults to ``dacbench/instance_sets/ga``.

    Returns:
    -------
    callable
        ``policy(env, state) -> int`` with a no-op ``policy.reset()``. The
        returned index respects ``env.action_choices``: a table entry whose
        ``choice`` is not in the environment's portfolio is mapped to the
        nearest value in log space.

    Raises:
    ------
    ValueError
        If the combination is one of :data:`UNVALIDATED_COMBINATIONS` and no
        explicit ``policy_path`` is given: no validated reference table
        ships for it. The derived policy was not at most 1.05 times the best
        fixed action on both paired seed sets A and B (OneMax), or the
        fitness-only bucket abstraction is insufficient (LeadingOne: the
        suffix after the leading ones drives mutation and crossover and is not
        part of ``f(x)``). Use ``success_rule`` or the best fixed action
        instead, or pass the path of a table that was generated with
        ``derive_optimal_policy.py --all``.
    FileNotFoundError
        If the table of a validated combination is missing on disk.

    """
    # the combination is fully explicit: fail as early as possible
    explicit = problem is not None and n is not None and algorithm is not None
    if policy_path is None and explicit and not is_validated(problem, n, algorithm):
        raise _unvalidated_error(_problem_stem(problem), n, algorithm)

    cache = {}

    def resolve(env):
        problem_name = problem
        if problem_name is None:
            problem_name = getattr(getattr(env, "problem", None), "__name__", None)
            if problem_name not in PROBLEM_FILE_STEMS:
                raise ValueError(
                    "Error: cannot infer the problem from the environment; pass "
                    "problem='OneMax' or problem='LeadingOne' explicitly"
                )
        size = n if n is not None else getattr(env, "n", None)
        if size is None:
            raise ValueError(
                "Error: cannot infer n from the environment before the first "
                "reset(); pass n explicitly or call the policy inside an episode"
            )
        alg = algorithm if algorithm is not None else getattr(env, "algorithm", None)
        if alg not in ALGORITHMS:
            raise ValueError(
                f"Error: unknown algorithm {alg!r}; pass one of {ALGORITHMS}"
            )
        key = (_problem_stem(problem_name), int(size), alg, str(policy_path))
        if key not in cache:
            path = _policy_path(problem_name, size, alg, policy_path, instance_dir)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Error: no derived policy table {path} for problem "
                    f"{problem_name!r}, n={size} and algorithm {alg!r}. Run "
                    "dacbench/instance_sets/ga/derive_optimal_policy.py first or "
                    "pass policy_path explicitly."
                )
            cache[key] = _build_lookup(load_policy_table(path))
        return cache[key]

    def policy(env, state):
        # API: env.n, env.x.fitness, env.algorithm, env.problem, env.action_choices
        lookup, buckets = resolve(env)
        _action, choice, _ = _best_row(lookup, buckets, int(env.x.fitness))
        return _map_choice_to_index(env, choice)

    policy.reset = lambda: None
    return policy


def best_static_portfolio(problem, n, algorithm, policy_path=None, instance_dir=None):
    """Return the sorted portfolio (``choice`` values) of a derived table.

    Useful to configure an environment with the exact portfolio a derived
    table refers to, so that the baselines and the reference policy see the same
    actions.

    Raises:
    ------
    ValueError
        If the combination has no validated table and no explicit
        ``policy_path`` is given (same rule as :func:`best_static`).

    """
    path = _policy_path(problem, n, algorithm, policy_path, instance_dir)
    df = load_policy_table(path)
    choices = np.sort(df["choice"].dropna().unique())
    if np.all(np.equal(np.mod(choices, 1), 0)):
        return [int(c) for c in choices]
    return [float(c) for c in choices]


def success_rule(algorithm, action_choices=None, initial_value=1, factor=1.5):
    """Return a stateful one-fifth success rule policy.

    The rule is the multiplicative one-fifth success rule (update strength
    ``factor = 1.5``). ``one_plus_one`` controls the mutation strength ``r``;
    the population-based algorithms control ``lambda``:

    - ``one_plus_one``: after an improvement ``r <- r * factor``, otherwise
      ``r <- r * factor**(-1/4)``. This is the multiplicative form of the
      one-fifth success rule of Kern et al. (2004, Eq. 3) as tabulated by Auger
      (GECCO 2009, Table 1); the original rule is due to Rechenberg (1973),
      see the module docstring for the references. Applying it to the integer
      mutation strength ``r`` of the (1+1) EA is the standard discrete analogue.
    - ``one_plus_lambda_lambda``: after an improvement ``lambda <- lambda /
      factor``, otherwise ``lambda <- lambda * factor**(1/4)``. This is the
      update of the self-adjusting (1+(lambda,lambda)) GA from Doerr & Doerr,
      GECCO 2015 (arXiv:1504.03212) and Algorithmica 2018
      (DOI 10.1007/s00453-017-0354-9). The direction is reversed with respect
      to the step-size rule because a larger population increases the success
      probability of an iteration.
    - ``one_plus_lambda``: the same lambda-update, labelled as an *adaptation*:
      the published one-fifth-rule analysis covers the (1+(lambda,lambda)) GA,
      not the (1+lambda) EA. The direction is the same for the same reason
      (larger lambda -> higher success probability). No theorem is claimed.

    The tracked value is mapped to the nearest action index in log space (the
    portfolio values are powers of two), clipped to the portfolio's range. With
    ``factor = 1.5`` the mean update is neutral at a success rate of 1/5, hence
    the name.

    The rule is stateful: it compares the fitness at each call with the fitness
    at the previous call (one call per environment step, and the environment
    accepts the offspring on ties, so fitness never decreases within an
    episode). It resets to ``initial_value`` whenever a new episode starts,
    detected robustly by ``env.c_step == 0`` or by a fitness *lower* than the
    last observed fitness (which cannot happen within an episode). The last
    step of an episode is not used for an update because the policy is not
    called after it, which does not affect the next episode.

    Parameters
    ----------
    algorithm : str
        One of :data:`ALGORITHMS`; selects what is adapted and in which
        direction.
    action_choices : sequence | None
        Portfolio of admissible values. Defaults to ``env.action_choices`` at
        call time. If the environment has no portfolio (continuous action
        space, only ``one_plus_one``), the adapted ``r`` is returned directly as
        an integer clipped to ``[1, n]``.
    initial_value : int | float
        Value the rule starts each episode with (default 1).
    factor : float
        Update strength ``F > 1`` (default 1.5).

    Returns:
    -------
    callable
        ``policy(env, state) -> int`` with ``policy.reset()``.

    """
    if algorithm not in ALGORITHMS:
        raise ValueError(
            f"Error: unknown algorithm {algorithm!r}; expected one of {ALGORITHMS}"
        )
    if factor <= 1:
        raise ValueError(f"Error: factor must be > 1, got {factor}")
    if action_choices is not None:
        action_choices = np.asarray(action_choices, dtype=float)
        if action_choices.size == 0:
            raise ValueError("Error: action_choices must not be empty")

    state_ = {"value": float(initial_value), "last_fitness": None}

    def reset():
        state_["value"] = float(initial_value)
        state_["last_fitness"] = None

    def policy(env, state):
        # API: env.c_step, env.x.fitness, env.action_choices, env.n
        choices = action_choices
        if choices is None:
            env_choices = getattr(env, "action_choices", None)
            if env_choices is not None:
                choices = np.asarray(env_choices, dtype=float)

        fitness = int(env.x.fitness)
        fresh_episode = (
            getattr(env, "c_step", 0) == 0
            or state_["last_fitness"] is None
            or fitness < state_["last_fitness"]
        )
        if fresh_episode:
            reset()
        elif algorithm == "one_plus_one":
            # step-size rule: grow on success, shrink by factor**(-1/4) otherwise
            state_["value"] *= (
                factor if fitness > state_["last_fitness"] else (factor**-0.25)
            )
        else:
            # population-size rule: shrink on success, grow by factor**(1/4)
            state_["value"] *= (
                factor**-1 if fitness > state_["last_fitness"] else (factor**0.25)
            )
        state_["last_fitness"] = fitness

        value = state_["value"]
        if choices is None:
            n = getattr(env, "n", None)
            value = value if n is None else float(np.clip(value, 1, n))
            state_["value"] = value
            return round(value)
        value = float(np.clip(value, choices.min(), choices.max()))
        state_["value"] = value
        with np.errstate(divide="ignore"):
            log_choices = np.log(choices)
        index = int(np.argmin(np.abs(log_choices - np.log(value))))
        return int(np.clip(index, 0, len(choices) - 1))

    policy.reset = reset
    return policy


def random_policy(seed=0):
    """Return a uniform random action policy.

    Parameters
    ----------
    seed : int
        Seed of the policy's own generator (independent of the environment).

    Returns:
    -------
    callable
        ``policy(env, state) -> int`` returning a uniform index in
        ``[0, len(env.action_choices))``. The policy is stateful through its own
        generator: call ``policy.reset()`` at the start of every episode (as
        ``run_episode`` in ``examples/run_ga_baselines.py`` does), otherwise the
        action stream continues across episodes and two runs are only
        comparable if they also reset identically. ``policy.reset()`` re-seeds
        the generator to ``seed``.

    """
    rng = np.random.default_rng(seed)

    def reset():
        nonlocal rng
        rng = np.random.default_rng(seed)

    def policy(env, state):
        # API: env.action_choices
        n_actions = len(env.action_choices)
        return int(rng.integers(n_actions))

    policy.reset = reset
    return policy
