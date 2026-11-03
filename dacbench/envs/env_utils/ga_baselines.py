"""Baseline and reference policies for the GA benchmark.

The GA benchmark (:mod:`dacbench.envs.ga`) controls the mutation strength ``r`` of
the (1+1) EA or the offspring population size ``lambda`` of the (1+lambda) EA and
the (1+(lambda,lambda)) GA. This module provides three policies that are useful as
reference points for DAC methods:

``best_static``
    Looks up the current fitness bucket ``f(x)`` in a reference table and
    returns the action with the smallest expected number of evaluations in that
    bucket. Tables ship for 10 of the 18 (problem, n, algorithm) combinations,
    six of them **exact** (OneMax with the (1+1) EA and the (1+lambda) EA); for
    the other eight it raises a ``ValueError`` because no validated table exists
    (see below).
``success_rule``
    The classic one-fifth success rule, applied either to ``r`` (``one_plus_one``)
    or to ``lambda`` (the population-based algorithms); for the (1+lambda) EA
    the lambda-update is an *adaptation* without a published analysis.
``random_policy``
    Uniform random action, the trivial baseline.

On top of these three policies the module exposes the optimality-gap API for the
exact OneMax combinations: :func:`optimal_reference` returns the exact optimal
evaluation count ``V[start_f]`` and the best fixed action, :func:`gap` turns one
episode into ``evals / optimal_evals`` (and refuses to do so for a censored or
non-exact run), and :func:`theory_policy` returns the theory-derived reference
policy of a combination. The four reference tiers, best first, are: exact
optimum (six OneMax combinations) > simulation-derived best known
``mean_evals`` +/- CI (four combinations) > measured best fixed action (all
combinations) > theory baseline.

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
A table ships for 10 of these 18 combinations: OneMax n=50, 100 and 200 with the
(1+1) EA and with the (1+lambda) EA (exact), OneMax n=100 and n=200 with the
(1+(lambda,lambda)) GA, LeadingOne n=100 with the (1+1) EA and the
(1+(lambda,lambda)) GA (those four simulation-derived). A table ships only if
the policy it encodes needs at most 1.05 times the evaluations of the best fixed
action on **both** paired seed sets of the validation (A: seeds 0-11, the
selection protocol; B: the held-out seeds 1000-1011) and is never significantly
worse -- which the six exact tables satisfy by construction, because their
policy *is* the optimum and can never be worse than a constant action. The
other eight have no validated table, because there the simulation-derived
policy was not demonstrably better than the best fixed action (see
:data:`UNVALIDATED_COMBINATIONS`); :func:`best_static` raises a ``ValueError``
for them, and ``derive_optimal_policy.py --all`` can still generate them for
research. The columns are, in this order::

    f_x,action,choice,mean_evals,n_seeds,ci_low,ci_high[,exact]

One row per (fitness bucket ``f_x``, portfolio index ``action``), sorted by
``f_x`` and then ``action``. ``choice`` is the parameter value (``r`` or
``lambda``) of that portfolio entry and ``mean_evals`` is the expected number of
evaluations needed to reach the optimum when running the bucket's fixed action
once and the table's policy from there on. The trailing ``exact`` column is
optional and only present in the exact tables (see below); ``best_static``
ignores it. ``mean_evals`` may be the literal ``inf``: in a simulation-derived
table no simulated sample of that action improved in that bucket, so the
improvement probability of the sample is zero and no finite estimate exists; in
an exact table it means that the action provably cannot leave the bucket. Such
rows are never selected.

Two kinds of tables
-------------------
**Exact (``exact = 1``).** For OneMax, ``f(x) = sum_i x_i`` is symmetric, so the
number of ones is a sufficient statistic: the successor distribution of a
generation depends on ``f(x)`` and the action only, and it is known in closed
form (a hypergeometric number of flipped ones). ``exact_optimal_policy.py``
evaluates it and runs the backward dynamic programme exactly. Its ``mean_evals``
is therefore the true optimal expected number of evaluations of the bucket
model -- for these operators, of the algorithm -- and the table has
``n_seeds = 0``, ``ci_low = ci_high = mean_evals`` and ``exact = 1``. The six
exact tables are OneMax n in ``{50, 100, 200}`` with the (1+1) EA and the
(1+lambda) EA (:data:`EXACT_COMBINATIONS`). For the (1+lambda) EA the exact
optimum happens to be the constant action ``lambda = 1`` in every bucket, so its
optimality gap is 1 by definition; the (1+1) EA gains 7-10 % over its best fixed
action (4 distinct actions, a large ``r`` far from the optimum, ``r = 1`` near
it).

**Simulation-derived.** For the four remaining shipped tables the bucket model
itself is not exact (LeadingOne's suffix, the crossover of the
(1+(lambda,lambda)) GA) and the numbers are Monte-Carlo estimates. For every
fitness bucket ``f(x)`` and every portfolio action, one generation is simulated
many times with common random numbers (the samples of all actions of a bucket
are paired), which estimates the expected evaluations per improvement and the
distribution of the successor bucket. A backward dynamic programme from
``f(x) = n`` (cost 0) then computes the expected cost to the optimum,
``V(b, a) = E[e] / q + E[V(b') | b' > b]`` with ``q = P(b' > b)``, if action
``a`` is played in bucket ``b`` and the derived policy afterwards. The CSV
column ``mean_evals`` holds this ``V(b, a)``. In both kinds of table
``best_static`` returns the per-bucket ``argmin_a V(b, a)``, i.e. exactly the
action the dynamic programme selects. This follows the methodology of
Biedenkapp, Dang, Krejca, Hutter & Doerr, "Theory-inspired parameter control
benchmarks for dynamic algorithm configuration", GECCO 2022,
arXiv:2202.03259, DOI 10.1145/3512290.3528846, who derive optimal portfolio-based
control policies for the mutation rate of RLS on LeadingOnes.

The simulation-derived tables are *not* backed by a theorem: they depend on the
fitness-bucket abstraction (the state is only ``f(x)``, so bit-level
correlations from the history are ignored), and the estimates carry Monte-Carlo
error (``n_seeds``, ``ci_low``, ``ci_high``). For those four combinations the
derived policy is the argmin of a *simulated* model only: these are
**approximate reference policies, not optima**, and their measured advantage
over the best fixed action is modest (0.79-1.01 times its evaluations). The six
exact tables are optima of the bucket model -- which, unlike the simulation
derived ones, is not an approximation for OneMax with these operators.

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

from math import sqrt
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

#: The six combinations whose shipped table is **exact** (``exact = 1``): the
#: table holds the true optimum of the bucket model, which for OneMax with these
#: operators is the true optimum of the algorithm *within the action portfolio*
#: (:data:`_PORTFOLIO`).  It is not the global optimum over all mutation
#: strengths: with ``r = n`` the (1+1) EA complements the string and solves
#: ``f(x) = 0`` in one evaluation, which the shipped portfolio excludes.  They
#: are produced by
#: ``exact_optimal_policy.py``, which refuses every other combination because the
#: fitness bucket is not a sufficient statistic there (LeadingOne's suffix, the
#: crossover of the (1+(lambda,lambda)) GA).
EXACT_COMBINATIONS = tuple(
    ("onemax", n, algorithm)
    for n in (50, 100, 200)
    for algorithm in ("one_plus_one", "one_plus_lambda")
)

#: Sizes for which reference data (exact or simulation-derived) exists.
REFERENCE_SIZES = (50, 100, 200)

#: File with the exact cost of every fixed action and of the optimal policy.
EXACT_FIXED_ACTIONS_FILE = "exact_fixed_action_counts.csv"

#: Row selector of the optimal cost ``V[start_f]`` in the fixed-action table.
#: Rows with ``action >= 0`` hold ``W_a[start_f]``, the cost of *always*
#: playing action ``a``.
EXACT_OPTIMAL_ACTION = -1

#: Columns the fixed-action table must contain.
_FIXED_COLUMNS = ("n", "algorithm", "start_f", "action", "choice", "expected_evals")

#: Module-level cache of the exact fixed-action data, keyed by resolved directory.
#: The table is read once per directory; later calls only index numpy arrays.
_EXACT_REFERENCE_CACHE = {}

#: The four shipped combinations whose table is simulation-derived and therefore
#: approximate: ``derive_optimal_policy.py`` produces them.
SIMULATED_COMBINATIONS = (
    ("onemax", 100, "one_plus_lambda_lambda"),
    ("onemax", 200, "one_plus_lambda_lambda"),
    ("leadingones", 100, "one_plus_one"),
    ("leadingones", 100, "one_plus_lambda_lambda"),
)

#: The ten (problem, n, algorithm) combinations whose table ships: the six exact
#: ones plus the four simulation-derived ones.  The simulation-derived ones ship
#: because their policy was at most 1.05 times the best fixed action on *both*
#: paired seed sets (A and B) and never significantly worse; the exact tables
#: satisfy that criterion by construction, because their policy is the optimum.
SHIPPED_COMBINATIONS = EXACT_COMBINATIONS + SIMULATED_COMBINATIONS

#: The eight combinations without a validated table. Every entry gives the
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
    """Return whether a validated reference table ships for a combination."""
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


def is_exact_table(path):
    """Return whether a CSV on disk is an exact table (``exact = 1``).

    Exact tables carry a trailing ``exact`` column that the simulation-derived
    tables do not have; ``derive_optimal_policy.py`` uses this to never replace
    one with a simulation estimate by accident.
    """
    path = Path(path)
    if not path.is_file():
        return False
    if "exact" not in pd.read_csv(path, nrows=1).columns:
        return False
    marks = pd.read_csv(path, usecols=["exact"])["exact"]
    return bool((pd.to_numeric(marks, errors="coerce") == 1).any())


def _policy_path(problem, n, algorithm, policy_path, instance_dir):
    """Return the path of the derived table for one (problem, n, algorithm).

    Raises:
    ------
    ValueError
        If the combination has no validated table and no explicit path is given.
    """
    if policy_path is not None:
        return Path(policy_path)
    if not is_validated(problem, n, algorithm):
        raise _unvalidated_error(_problem_stem(problem), n, algorithm)
    directory = Path(instance_dir) if instance_dir is not None else DEFAULT_INSTANCE_DIR
    return directory / policy_file_name(problem, n, algorithm)


def load_policy_table(path):
    """Read and validate a reference table (exact or simulation-derived).

    ``mean_evals`` is coerced to float; the literal ``inf`` (and any value that
    cannot be parsed as a number) marks an action that never improved, and such
    a row is therefore never selected. The optional ``exact`` column of the
    exact tables (see :data:`EXACT_COMBINATIONS`) is carried through unchanged
    and not used for the selection. The returned frame is sorted by ``f_x`` and
    ``action``.

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
    table = pd.read_csv(path)
    missing = [c for c in _REQUIRED_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(
            f"Error: {path} is missing the column(s) {missing}; expected at least "
            f"{list(_REQUIRED_COLUMNS)}"
        )
    table = table.copy()
    table["f_x"] = pd.to_numeric(table["f_x"], errors="raise").astype(float)
    table["action"] = pd.to_numeric(table["action"], errors="raise").astype(int)
    table["choice"] = pd.to_numeric(table["choice"], errors="raise").astype(float)
    # "inf" parses directly; anything unparsable becomes NaN -> treated as inf
    table["mean_evals"] = pd.to_numeric(table["mean_evals"], errors="coerce").astype(
        float
    )
    return table.sort_values(["f_x", "action"]).reset_index(drop=True)


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


def _map_choice_to_index(env, choice, choices=None):
    """Map a portfolio value to the matching action index of ``env``.

    If the value is listed, its index is returned; otherwise the nearest value in
    log space is used (portfolio values are powers of two, so log space is the
    natural metric).  ``choices`` defaults to ``env.action_choices``; an
    environment without that list (continuous action space, only
    ``one_plus_one``) receives the raw value rounded to an integer.
    """
    if choices is None:
        choices = getattr(env, "action_choices", None)
    if choices is None:
        # continuous action space (only ``one_plus_one``): the table values are
        # integers, so the raw value is the action
        return round(float(choice))
    choices = np.asarray(choices, dtype=float)
    match = np.flatnonzero(np.isclose(choices, choice))
    if match.size:
        return int(match[0])
    with np.errstate(divide="ignore"):
        log_choices = np.log(choices)
    index = int(np.argmin(np.abs(log_choices - np.log(choice))))
    return int(np.clip(index, 0, len(choices) - 1))


def _validate_reference_size(n):
    """Return ``n`` as an int if reference data exists for that size."""
    if isinstance(n, bool) or not isinstance(n, int | np.integer):
        raise ValueError(
            f"Error: n must be an integer, got {n!r}; reference data exists for "
            f"n in {REFERENCE_SIZES}"
        )
    size = int(n)
    if size not in REFERENCE_SIZES:
        raise ValueError(
            f"Error: unknown n {size}; reference data exists for n in {REFERENCE_SIZES}"
        )
    return size


def _validate_algorithm(algorithm):
    """Return ``algorithm`` if it is one of :data:`ALGORITHMS`."""
    if algorithm not in ALGORITHMS:
        raise ValueError(
            f"Error: unknown algorithm {algorithm!r}; expected one of {ALGORITHMS}"
        )
    return algorithm


def _validate_start_f(start_f, n):
    """Return the start bucket as an integer in ``[0, n]`` (``None`` means 0)."""
    if start_f is None:
        return 0
    if isinstance(start_f, bool) or not isinstance(start_f, int | np.integer):
        raise ValueError(
            f"Error: start_f must be an integer in [0, {n}], got {start_f!r}"
        )
    value = int(start_f)
    if not 0 <= value <= n:
        raise ValueError(
            f"Error: start_f must be an integer in [0, {n}], got {start_f!r}"
        )
    return value


def _not_exact_reason(problem_stem, algorithm):
    """Return why the bucket model is not exact for a combination.

    The reasons are structural and do not depend on the size: they name the
    part of the bitstring that the fitness ``f(x)`` does not determine.
    """
    if problem_stem == "leadingones":
        return (
            "LeadingOne is not covered by an exact bucket model: the suffix "
            "after the leading ones is not part of f(x), but it drives mutation "
            "and crossover, so f(x) is not a sufficient statistic"
        )
    return (
        "the (1+(lambda,lambda)) GA is not covered by an exact bucket model: "
        "the crossover child's fitness and its evaluation count "
        "(count_different_inds_only) depend on the arrangement of the bits, not "
        "only on f(x), so f(x) is not a sufficient statistic"
    )


def _exact_reference_data(instance_dir=None):
    """Read ``exact_fixed_action_counts.csv`` once and index it by combination.

    The returned mapping is keyed by ``(n, algorithm)`` and holds, for every
    start bucket ``f = 0..n``, the exact optimal cost ``V[f]``, the exact cost
    ``W_a[f]`` of always playing every portfolio action, and the portfolio
    values. The parser checks the documented relation
    ``V[f] <= min_a W_a[f]``: it is an equality when the optimal policy is a
    constant action -- in particular for the whole (1+lambda) EA -- and strict
    when adapting after the first generation helps, as it does for the (1+1) EA
    on the lower half of the buckets.  ``instance_dir`` defaults to
    :data:`DEFAULT_INSTANCE_DIR`; the resolved directory is the cache key.
    """
    directory = Path(instance_dir) if instance_dir is not None else DEFAULT_INSTANCE_DIR
    key = directory.resolve()
    if key in _EXACT_REFERENCE_CACHE:
        return _EXACT_REFERENCE_CACHE[key]

    path = directory / EXACT_FIXED_ACTIONS_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"Error: exact fixed-action table not found: {path}. Run "
            "dacbench/instance_sets/ga/exact_optimal_policy.py to generate it."
        )
    table = pd.read_csv(path)
    missing = [column for column in _FIXED_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(
            f"Error: {path} is missing the column(s) {missing}; expected at "
            f"least {list(_FIXED_COLUMNS)}"
        )

    data = {}
    for (n, algorithm), block in table.groupby(["n", "algorithm"], sort=True):
        block = block.copy()
        block["start_f"] = pd.to_numeric(block["start_f"], errors="raise").astype(int)
        block["action"] = pd.to_numeric(block["action"], errors="raise").astype(int)
        block["choice"] = pd.to_numeric(block["choice"], errors="raise").astype(float)
        block["expected_evals"] = pd.to_numeric(
            block["expected_evals"], errors="raise"
        ).astype(float)

        optimal = block[block["action"] == EXACT_OPTIMAL_ACTION].sort_values("start_f")
        fixed = block[block["action"] >= 0]
        pivot = fixed.pivot_table(
            index="action",
            columns="start_f",
            values="expected_evals",
            aggfunc="first",
        ).sort_index()
        actions = pivot.index.to_numpy(dtype=int)
        costs = pivot.to_numpy(dtype=float)
        values = optimal["expected_evals"].to_numpy(dtype=float)
        starts = optimal["start_f"].to_numpy(dtype=int)
        if starts.tolist() != list(range(int(n) + 1)):
            raise ValueError(
                f"Error: {path} does not hold the optimal cost for every start "
                f"bucket 0..{int(n)} of n={int(n)}, algorithm {algorithm!r}"
            )
        # V[f] <= min_a W_a[f] always: a policy that may switch after the first
        # generation cannot be worse than always playing one action.  Only the
        # constant-optimum case (the whole (1+lambda) EA) makes it an equality.
        tolerance = 1e-9 * np.maximum(1.0, costs.min(axis=0))
        if np.any(values - costs.min(axis=0) > tolerance):
            raise ValueError(
                f"Error: {path} is inconsistent for n={int(n)}, algorithm "
                f"{algorithm!r}: V[start_f] must not exceed min_a W_a[start_f]"
            )
        choice_by_action = fixed.drop_duplicates("action").set_index("action")["choice"]
        choices = choice_by_action.reindex(actions).to_numpy(dtype=float)
        data[(int(n), algorithm)] = {
            "optimal": values,
            "fixed": costs,
            "actions": actions,
            "choices": choices,
        }

    _EXACT_REFERENCE_CACHE[key] = data
    return data


def optimal_reference(problem, n, algorithm, start_f=None, *, instance_dir=None):
    """Return the exact optimal evaluation count of a combination, if one exists.

    For the six exact combinations (:data:`EXACT_COMBINATIONS`) the returned
    numbers are the true optimum of the bucket model, which for OneMax with
    these mutation-only operators is the true optimum of the algorithm *within
    the action portfolio*: ``V`` and ``W`` come from the backward dynamic
    programme of ``exact_optimal_policy.py``, not from sampling. It is not the
    global optimum over all mutation strengths: with ``r = n`` the (1+1) EA
    complements the string and solves ``f(x) = 0`` in one evaluation, which the
    shipped portfolio excludes, so a policy using such actions is outside the
    reference's action set and its gap is not comparable.

    For every other combination the function returns ``available = False`` and
    a reason. It never returns an approximate number in ``optimal_evals``: an
    approximate table is a best-known reference, not an optimum.

    Parameters
    ----------
    problem : str
        ``OneMax`` / ``LeadingOne`` or their file-name stems.
    n : int
        Problem size, one of :data:`REFERENCE_SIZES`.
    algorithm : str
        One of :data:`ALGORITHMS`.
    start_f : int | None
        Number of ones of the initial solution, ``0 <= start_f <= n``.
        ``None`` (default) means the hardest start, ``0``.
    instance_dir : str | Path | None
        Directory with ``exact_fixed_action_counts.csv``; defaults to
        :data:`DEFAULT_INSTANCE_DIR`.

    Returns:
    -------
    dict
        For an exact combination::

            {"available": True, "exact": True, "optimal_evals": float,
             "best_fixed_action": int, "best_fixed_choice": int | float,
             "best_fixed_evals": float, "start_f": int, "source": "exact DP"}

        and otherwise::

            {"available": False, "exact": False, "optimal_evals": None,
             "reason": str}

        ``optimal_evals`` is ``V[start_f]``, the exact optimal expected number
        of evaluations to reach the optimum, not counting the initial
        individual. ``best_fixed_evals`` is the smallest ``W_a[start_f]``, the
        exact cost of always playing one portfolio action. ``V <= W``, with
        equality exactly when the optimal policy is a constant action (always
        for the (1+lambda) EA, and in the upper buckets of the (1+1) EA).

    Raises:
    ------
    ValueError
        If the problem, the size, the algorithm or ``start_f`` is invalid. The
        message lists the supported values.
    FileNotFoundError
        If an exact combination is requested but the fixed-action table is
        missing.

    """
    problem_stem = _problem_stem(problem)
    _validate_algorithm(algorithm)
    size = _validate_reference_size(n)
    start = _validate_start_f(start_f, size)

    if (problem_stem, size, algorithm) not in EXACT_COMBINATIONS:
        return {
            "available": False,
            "exact": False,
            "optimal_evals": None,
            "reason": _not_exact_reason(problem_stem, algorithm),
        }

    data = _exact_reference_data(instance_dir)[(size, algorithm)]
    costs = data["fixed"][:, start]
    best_index = int(np.argmin(costs))
    choice = data["choices"][best_index]
    best_choice = int(choice) if float(choice).is_integer() else float(choice)
    return {
        "available": True,
        "exact": True,
        "optimal_evals": float(data["optimal"][start]),
        "best_fixed_action": int(data["actions"][best_index]),
        "best_fixed_choice": best_choice,
        "best_fixed_evals": float(costs[best_index]),
        "start_f": start,
        "source": "exact DP",
    }


def gap(evals, problem, n, algorithm, start_f, *, solved=True, instance_dir=None):
    """Return the optimality gap of one episode against the exact reference.

    ``ratio = evals / optimal_evals``. A ratio of 1 means the policy matched
    the exact optimum for the start bucket; 1.15 means it needed 15 % more
    evaluations. The reference is per start bucket, so two episodes with
    different initial fitnesses are compared against their own optimum.

    A **censored** episode (``solved = False``, the evaluation cutoff was hit
    before the optimum) gets no ratio: its ``evals`` is only a lower bound, and
    reporting ``evals / optimal_evals`` would present an unsolved run as a small
    gap. For non-exact combinations there is no reference, so ``ratio`` is
    ``None`` as well and ``reason`` explains why.

    Parameters
    ----------
    evals : int | float
        Evaluations spent on the search. ``GAEnv.total_evals`` also counts the
        initial individual (it is 1 after ``reset``); the reference excludes it,
        so pass ``env.total_evals - 1`` for a comparable number.
    problem : str
        ``OneMax`` / ``LeadingOne`` or their file-name stems.
    n : int
        Problem size, one of :data:`REFERENCE_SIZES`.
    algorithm : str
        One of :data:`ALGORITHMS`.
    start_f : int
        Number of ones of the initial solution.
    solved : bool
        Whether the episode reached the optimum before the evaluation cutoff.
    instance_dir : str | Path | None
        Directory with ``exact_fixed_action_counts.csv``.

    Returns:
    -------
    dict
        ``{"evals": float, "reference_evals": float | None,
        "ratio": float | None, "exact": bool, "solved": bool,
        "reason": str | None}``.

    Raises:
    ------
    ValueError
        For an invalid combination or ``start_f`` (see
        :func:`optimal_reference`).

    """
    evaluations = float(evals)
    reference = optimal_reference(
        problem, n, algorithm, start_f, instance_dir=instance_dir
    )
    result = {
        "evals": evaluations,
        "reference_evals": reference["optimal_evals"],
        "ratio": None,
        "exact": bool(reference["exact"]),
        "solved": bool(solved),
        "reason": None,
    }
    if not solved:
        result["reason"] = (
            "censored: the episode hit the evaluation cutoff before reaching "
            "the optimum, so evals is only a lower bound and no optimality gap "
            "is defined"
        )
        return result
    if not reference["available"]:
        result["reference_evals"] = None
        result["reason"] = reference["reason"]
        return result
    if reference["optimal_evals"] <= 0:
        result["reason"] = (
            "start_f = n: the initial solution is already optimal, so no "
            "evaluation is needed and no gap is defined"
        )
        return result
    result["ratio"] = evaluations / reference["optimal_evals"]
    return result


def theory_policy(problem, algorithm):
    """Return a theory-derived reference policy for one combination.

    The policy has the ``policy(env, state) -> int`` signature of the other
    baselines and returns an action **index** in
    ``[0, len(env.action_choices))``. It is stateless (``policy.reset()`` is a
    no-op). It reads only documented environment attributes: the current
    fitness ``env.x.fitness``, the problem size ``env.n`` and the portfolio
    ``env.action_choices``.

    This is a **theory-derived reference, not an optimum**, with two
    exceptions where the theory value coincides with the exact dynamic
    programme (:func:`optimal_reference`): the (1+1) EA and the (1+lambda) EA
    on OneMax. The four-tier hierarchy is: exact optimum > simulation-derived
    best known ``mean_evals`` +/- CI > measured best fixed action > this theory
    baseline.

    The policies:

    - ``one_plus_one`` on OneMax: constant ``r = 1``, i.e. mutation rate
      ``p = 1/n``. This is the optimal mutation rate for the (1+1) EA on OneMax
      and linear functions (see Doerr's survey, arXiv:1801.04487, Section 6;
      Witt, Sudholt), and it is exactly the exact-DP optimum at this start.
    - ``one_plus_lambda`` on OneMax: constant ``lambda = 1``. The exact DP of
      ``exact_fixed_action_counts.csv`` proves that this constant action is the
      optimum in every bucket: a generation with ``lambda`` children costs
      ``lambda`` times as much and improves ``lambda`` times as often, so
      parallelism does not buy evaluations on OneMax.
    - ``one_plus_lambda_lambda`` on OneMax: the fitness-dependent
      ``lambda = round(sqrt(n / max(1, n - f(x))))`` of Doerr, Doerr & Ebel
      (TCS 2015), used as the ``theory`` baseline in
      Chen, Buzdalov, Doerr & Dang (arXiv:2302.12334). The value is mapped to
      the nearest portfolio value in log space. If it exceeds the largest
      portfolio value it is clipped to that value: the formula grows like
      ``sqrt(n)`` near the optimum, which is above the default portfolio at
      large ``n``.
    - ``one_plus_one`` on LeadingOne: constant ``r = 2``, the nearest portfolio
      value to the best known fixed mutation rate ``p ~ 1.59/n`` for the (1+1)
      EA on LeadingOnes (Boettcher, Doerr & Neumann, PPSN 2010; Doerr's survey).
      This is a theory-derived constant, not an adaptation.
    - ``one_plus_lambda`` on LeadingOne: constant ``lambda = 1``, the nearest
      portfolio value to the same fixed mutation rate (the (1+lambda) EA uses
      ``p = 1/n``; ``lambda = 1`` is the smallest admissible population).
    - ``one_plus_lambda_lambda`` on LeadingOne: raises ``ValueError``. There is
      no published analysis of the (1+(lambda,lambda)) GA on LeadingOnes, and
      the exact bucket model is not available because the suffix after the
      leading ones is not part of ``f(x)``.

    Parameters
    ----------
    problem : str
        ``OneMax`` / ``LeadingOne`` or their file-name stems.
    algorithm : str
        One of :data:`ALGORITHMS`.

    Returns:
    -------
    callable
        ``policy(env, state) -> int`` with a no-op ``policy.reset()``.

    Raises:
    ------
    ValueError
        If the problem or the algorithm is unknown, or if no theory policy is
        available for the combination (the (1+(lambda,lambda)) GA on
        LeadingOne).

    """
    problem_stem = _problem_stem(problem)
    _validate_algorithm(algorithm)
    if problem_stem == "leadingones" and algorithm == "one_plus_lambda_lambda":
        raise ValueError(
            "Error: no theory policy for the (1+(lambda,lambda)) GA on "
            "LeadingOne: there is no published runtime analysis of that "
            "operator on LeadingOnes, and the exact fitness-bucket model is "
            "not available because the suffix after the leading ones is not "
            "part of f(x). Use success_rule or the measured best fixed action."
        )

    # constant action value, or ``None`` for the fitness-dependent GA rule
    constant = {
        ("onemax", "one_plus_one"): 1.0,
        ("onemax", "one_plus_lambda"): 1.0,
        ("onemax", "one_plus_lambda_lambda"): None,
        ("leadingones", "one_plus_one"): 2.0,
        ("leadingones", "one_plus_lambda"): 1.0,
    }[(problem_stem, algorithm)]

    def policy(env, state):
        # API: env.n, env.x.fitness, env.action_choices
        if constant is not None:
            return _map_choice_to_index(env, constant)
        n = int(env.n)
        fitness = int(env.x.fitness)
        value = round(sqrt(n / max(1, n - fitness)))
        return _map_choice_to_index(env, value)

    policy.reset = lambda: None
    return policy


def best_static(
    problem=None, n=None, algorithm=None, policy_path=None, instance_dir=None
):
    """Return the per-bucket greedy policy of a reference table.

    The policy looks up the current fitness ``f(x)`` in the table identified by
    (problem, n, algorithm) and returns the action with the smallest finite
    ``mean_evals`` (the cost-to-go ``V(f(x), a)``, exact for the exact tables and
    simulated for the others) for that bucket, i.e. the action of the dynamic
    programme that produced the table. Buckets that are not in the table (gaps,
    including ``f(x)`` values above the largest bucket and the optimum) are
    resolved to the nearest bucket; ``inf`` rows are never selected.

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
        ships for it. The simulation-derived policy was not at most 1.05 times
        the best fixed action on both paired seed sets A and B (OneMax), or the
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
    table = load_policy_table(path)
    choices = np.sort(table["choice"].dropna().unique())
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
        return _map_choice_to_index(env, value, choices)

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
