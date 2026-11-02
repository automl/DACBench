"""Exact optimal reference policies for the GA benchmark.

This is a *standalone* script.  It is never imported by the ``dacbench`` package
and no part of the package depends on it: the CSVs it produces are reference
data, not code.  Run it from a checkout, e.g.::

    python dacbench/instance_sets/ga/exact_optimal_policy.py

With no arguments it regenerates the six exact tables
``optimal_policy_onemax_{50,100,200}_{one_plus_one,one_plus_lambda}.csv`` and the
fixed-action table ``exact_fixed_action_counts.csv``, all inside this directory.

Unlike the simulation-based sibling script ``derive_optimal_policy.py``, the
numbers here are **not estimates**: for OneMax with the (1+1) EA and the
(1+lambda) EA, the fitness bucket ``f(x)`` (the number of ones) is a sufficient
statistic, so the transition probabilities of the bucket chain are known in
closed form and the backward dynamic programme computes the *exact* optimal
expected number of evaluations of every bucket and every action.  ``V[f]`` is
therefore a real optimum of the bucket model, and a policy derived from this
table has an optimality gap of exactly 1 (it *is* optimal), not a confidence
interval.

The model
---------
Notation: ``n`` bits, ``f = f(x)`` ones, one generation, elitist acceptance
``x <- child`` if ``f(child) >= f(x)``.

**Mutation.** Let ``L`` be the number of flipped bits of one child.  Both
operators of this script draw ``L ~ Binomial(n, p)`` conditioned on ``L > 0``
(resampling until ``L > 0`` is the ``_{>0}`` convention of the environment):

* ``one_plus_one``: one child, mutation rate ``p = r / n`` for the action value
  ``r``; the generation costs exactly 1 evaluation.
* ``one_plus_lambda``: the mutation rate is fixed at ``p = 1 / n`` and ``lambda``
  independent children are created, child ``i`` with its own
  ``L_i ~ Binomial(n, p) | L_i > 0``.  The generation costs exactly ``lambda``
  evaluations (``mutate_independent`` returns ``n_childs``).

**Fitness after flipping ``l`` distinct uniformly chosen bits.**  Of the ``l``
flipped positions, ``A`` carry a one:

    A ~ Hypergeometric(n, f, l) = C(f, a) * C(n - f, l - a) / C(n, l)

Flipping a one subtracts one from the fitness, flipping a zero adds one, so

    f' = f - A + (l - A) = f + l - 2A.

(``OneMax.get_fitness_after_flipping`` computes exactly this:
``f(x) + len(locs) - 2 * (number of ones among the flipped bits)``.)  Because
``A`` depends only on ``f`` and ``l``, and ``L`` only on the action, the
successor fitness distribution depends only on ``(f, action)``: the bucket is a
sufficient statistic.

**Per-generation transition matrix** (``T_a[f, g]``, rows ``f = 0..n``):

* ``one_plus_one``::

      T_r[f, g] = sum over (l, a) with max(f, f + l - 2a) = g of
                      P(L = l) * P(A = a | f, l)

* ``one_plus_lambda``: the children are iid, so with
  ``C_r[f, g] = sum_{h <= g} T_r[f, h]`` the single-child distribution for
  ``p = 1/n`` (the ``one_plus_one`` row with ``r = 1``)::

      T_lambda[f, g] = C[f, g] ** lambda - C[f, g - 1] ** lambda   for g > f
      T_lambda[f, f] = C[f, f] ** lambda

  i.e. ``P(max_i f'_i = g)`` for ``g >= f``.  Rows sum to one because
  ``C[f, n] = 1``.

Acceptance is elitist, so the self-loop mass ``T_a[f, f]`` (``g = f``) collects
every generation that does not improve, including generations that make the
child worse.  The bucket ``n`` is terminal (``T_a[n, n] = 1``).

**Numerics.**  Hypergeometric probabilities are evaluated in log space with
``math.lgamma`` and ``numpy`` broadcasting, which avoids the overflow that the
product of three binomial coefficients has in double precision for ``n`` far above
the shipped sizes; there is no dependency beyond ``numpy`` and ``pandas``.
The binomial probability of a flip count ``l`` that is below
``BINOMIAL_TRUNCATION = 1e-16`` is set to zero and every row of ``T`` is then
renormalised to sum to one, so the truncated tails are redistributed over the
remaining flip counts.  The dropped mass is at most a few ``1e-16`` per row.

Backward dynamic programme
--------------------------
With ``V[n] = 0`` and ``q_a(f) = 1 - T_a[f, f]`` (the probability that one
generation with action ``a`` leaves bucket ``f``), for ``f = n-1 ... 0``::

    V[f, a] = (c_a + sum_{g > f} T_a[f, g] * V[g]) / q_a(f)       (inf if q_a(f) == 0)
    V[f]    = min_a V[f, a]
    pi(f)   = argmin_a V[f, a]

``V[f, a]`` is the expected number of evaluations of "play ``a`` once, then
follow the optimal policy"; the division by ``q_a(f)`` resolves the self-loops
(``1 / q`` generations are needed per improvement).  This is the same convention
as the simulation-based script, so the ``mean_evals`` column of both kinds of
table means the same thing and ``best_static`` selects ``argmin_a`` in both.
``V[f]`` is the exact optimal expected number of evaluations to reach the
optimum from bucket ``f``, *not counting* the evaluation of the initial
individual (the environment's ``total_evals`` starts at 1).

For every fixed action the script also computes the exact cost of *always*
playing that action::

    W_a[n] = 0
    W_a[f] = (c_a + sum_{g > f} T_a[f, g] * W_a[g]) / q_a(f)       (inf if q_a(f) == 0)

Both tables are written with full double precision (the shortest
round-tripping representation), so ``argmin`` over a bucket is reproducible from
the CSV.

Where the bucket model is *not* exact (refused, not approximated)
-----------------------------------------------------------------
The script refuses to compute a table it cannot prove, and names the reason:

* ``one_plus_lambda_lambda`` on OneMax: the mutation phase flips the *same* ``l``
  bits in all ``lambda`` mutants, and the crossover phase recombines a mutant
  with the incumbent.  Whether a crossover offspring is better than both parents
  depends on the Hamming distance between them, not on ``f(x)`` alone: two
  bitstrings with the same fitness but a different distance to the mutant give
  different successor distributions.  ``f(x)`` is not a sufficient statistic, so
  a bucket table would be an approximation.
* every LeadingOne combination (all three algorithms): ``f(x) = b`` fixes the
  prefix ``0..b-1`` to one and bit ``b`` to zero, but not the suffix
  ``b+1..n-1``.  The successor fitness of a mutation (and much more of a
  crossover) depends on that suffix, so the bucket chain is not Markov.

Combinations that are refused raise :class:`NotExactError`, a ``ValueError``
that names the offending combination and the reason.  The script never emits a
table labelled ``exact = 1`` for them.

Output
------
* One table per (OneMax, n, algorithm) with ``algorithm`` in
  ``{one_plus_one, one_plus_lambda}``, named like the simulation-based tables
  (``optimal_policy_{problem}_{n}_{algorithm}.csv``) so that
  :func:`dacbench.envs.env_utils.ga_baselines.best_static` reads it unchanged.
  The columns are those of the simulated tables plus a trailing ``exact``
  column::

      f_x,action,choice,mean_evals,n_seeds,ci_low,ci_high,exact

  There is no sampling, so ``n_seeds = 0``, ``ci_low = ci_high = mean_evals`` and
  ``exact = 1`` on every row.  Rows are sorted by ``f_x`` and ``action``; the
  terminal bucket ``f_x = n`` is included with ``mean_evals = 0``.
  ``mean_evals`` is ``inf`` exactly where ``q_a(f) = 0``, i.e. where the action
  provably cannot leave the bucket.
* ``exact_fixed_action_counts.csv`` with the columns::

      n,algorithm,start_f,action,choice,expected_evals

  one row per (n, algorithm, start bucket ``0..n``, portfolio action) holding
  ``W_a[start_f]``, plus one row per (n, algorithm, start bucket) with
  ``action = -1, choice = 0`` holding the optimal ``V[start_f]``.  This is the
  table that makes exact comparisons against a constant action possible.

Determinism: the script has no random component, so the identical invocation
produces byte-identical CSVs.  ``--check`` recomputes everything and reports
whether the files on disk are up to date instead of writing them.
"""

from __future__ import annotations

import argparse
import time
from math import lgamma
from pathlib import Path

import numpy as np
import pandas as pd

PROBLEMS = ("onemax", "leadingones")
ALGORITHMS = ("one_plus_one", "one_plus_lambda", "one_plus_lambda_lambda")
DEFAULT_SIZES = (50, 100, 200)
DEFAULT_PORTFOLIO = (1, 2, 4, 8, 16)

#: Directory with the shipped instance sets.  It must be absolute:
#: ``GABenchmark`` looks up a same-named CSV in the current working directory
#: first, so a relative name would silently change behaviour.
INSTANCE_DIR = Path(__file__).resolve().parent

#: Problems for which the fitness bucket is a sufficient statistic.
EXACT_PROBLEMS = ("onemax",)

#: Algorithms for which the fitness bucket is a sufficient statistic.  The
#: (1+(lambda,lambda)) GA is missing on purpose, see :class:`NotExactError`.
EXACT_ALGORITHMS = ("one_plus_one", "one_plus_lambda")

#: Binomial probabilities below this value are dropped from the flip-count sum
#: and the remaining row is renormalised (see the module docstring).
BINOMIAL_TRUNCATION = 1e-16

#: File name of the fixed-action cost table.
FIXED_ACTIONS_FILE = "exact_fixed_action_counts.csv"

#: Columns of an exact policy table: the columns of the simulated tables plus
#: the ``exact`` marker the simulation script must not overwrite.
COLUMNS = (
    "f_x",
    "action",
    "choice",
    "mean_evals",
    "n_seeds",
    "ci_low",
    "ci_high",
    "exact",
)

#: Columns of :data:`FIXED_ACTIONS_FILE`.
FIXED_COLUMNS = ("n", "algorithm", "start_f", "action", "choice", "expected_evals")

#: Row selector of the optimal (not fixed-action) cost in the fixed-action table.
OPTIMAL_ACTION = -1


class NotExactError(ValueError):
    """Raised when the bucket model is not provably exact for a combination.

    The script refuses to compute such a table instead of labelling an
    approximation ``exact``.  A ``ValueError`` subclass so that callers can
    catch it like any other invalid-argument error.
    """


def _not_exact_reason(problem, algorithm):
    """Return why the bucket model is not exact, or ``None`` if it is exact.

    Parameters
    ----------
    problem : str
        Problem slug, ``onemax`` or ``leadingones``.
    algorithm : str
        One of ``ALGORITHMS``.

    Returns:
    -------
    str | None
        The reason for refusing the combination, ``None`` for an exact one.

    """
    if problem not in PROBLEMS:
        raise ValueError(f"Error: unknown problem {problem!r}; expected {PROBLEMS}")
    if algorithm not in ALGORITHMS:
        raise ValueError(
            f"Error: unknown algorithm {algorithm!r}; expected {ALGORITHMS}"
        )
    if problem not in EXACT_PROBLEMS:
        return (
            f"the fitness bucket is not a sufficient statistic for "
            f"{problem!r}: f(x) = b fixes the leading ones (prefix and the first "
            f"zero) but not the suffix after them, and the successor fitness of "
            f"a mutation (and of a crossover even more) depends on that suffix"
        )
    if algorithm not in EXACT_ALGORITHMS:
        return (
            f"the fitness bucket is not a sufficient statistic for "
            f"{algorithm!r}: the mutation phase flips the same l bits in all "
            f"lambda mutants and the crossover phase recombines a mutant with "
            f"the incumbent, so whether an offspring improves depends on the "
            f"Hamming distance between the parents, which f(x) does not "
            f"determine"
        )
    return None


def require_exact(problem, algorithm):
    """Raise :class:`NotExactError` unless the bucket model is exact.

    Parameters
    ----------
    problem : str
        Problem slug, ``onemax`` or ``leadingones``.
    algorithm : str
        One of ``ALGORITHMS``.

    Raises:
    ------
    NotExactError
        If the fitness bucket is not a sufficient statistic for the combination;
        the message names the combination and the reason.
    ValueError
        If the problem or the algorithm is unknown.

    """
    reason = _not_exact_reason(problem, algorithm)
    if reason is not None:
        raise NotExactError(
            f"Error: no exact table for problem {problem!r} and algorithm "
            f"{algorithm!r}: {reason}. Use derive_optimal_policy.py for a "
            f"simulation-based approximation instead."
        )


def output_name(problem, n, algorithm):
    """Return the CSV file name of the exact policy table of one combination.

    Parameters
    ----------
    problem : str
        Problem slug, ``onemax`` or ``leadingones``.
    n : int
        Problem size.
    algorithm : str
        Algorithm name.

    Returns:
    -------
    str
        File name of the exact policy table (the same scheme as the
        simulation-based tables, so that ``best_static`` finds it unchanged).

    """
    return f"optimal_policy_{problem}_{int(n)}_{algorithm}.csv"


def log_factorials(n):
    """Return ``log(k!)`` for ``k = 0..n`` as an array.

    Parameters
    ----------
    n : int
        Largest factorial argument.

    Returns:
    -------
    numpy.ndarray
        Array of length ``n + 1`` with ``log(k!)``; used to build binomial and
        hypergeometric log-probabilities without overflow.

    """
    return np.array([lgamma(k + 1.0) for k in range(n + 1)], dtype=float)


def flip_count_distribution(n, p, log_fact):
    """Return ``P(L = l)`` for ``l = 1..n`` conditioned on ``L > 0``.

    ``L ~ Binomial(n, p)`` is the number of flipped bits of one child, resampled
    until ``L > 0`` (the ``_{>0}`` convention of the environment).  Probabilities
    below :data:`BINOMIAL_TRUNCATION` are set to zero; the remaining entries are
    divided by ``1 - (1 - p)**n`` and therefore sum to one only up to the
    truncated mass, which is why the transition rows are renormalised later.

    Parameters
    ----------
    n : int
        Number of bits.
    p : float
        Mutation rate, ``0 < p <= 1``.
    log_fact : numpy.ndarray
        ``log(k!)`` for ``k = 0..n``, see :func:`log_factorials`.

    Returns:
    -------
    numpy.ndarray
        Array of length ``n`` with ``P(L = l)`` for ``l = 1..n``.

    """
    if p >= 1.0:
        # every bit is flipped: L = n with probability 1
        pmf = np.zeros(n)
        pmf[-1] = 1.0
        return pmf
    flips = np.arange(1, n + 1)
    q = 1.0 - p
    log_pmf = (
        log_fact[n]
        - log_fact[flips]
        - log_fact[n - flips]
        + flips * np.log(p)
        + (n - flips) * np.log(q)
    )
    pmf = np.exp(log_pmf)
    pmf[pmf < BINOMIAL_TRUNCATION] = 0.0
    return pmf / (1.0 - q**n)


def hypergeometric_matrix(n, f, log_fact):
    """Return ``P(A = a | f, l)`` for every flip count ``l`` and one count ``a``.

    ``A`` is the number of ones among ``l`` distinct uniformly chosen bits of a
    string with ``f`` ones (hypergeometric distribution).

    Parameters
    ----------
    n : int
        Number of bits.
    f : int
        Number of ones of the parent, ``0 <= f < n``.
    log_fact : numpy.ndarray
        ``log(k!)`` for ``k = 0..n``, see :func:`log_factorials`.

    Returns:
    -------
    numpy.ndarray
        Array of shape ``(n, f + 1)``; entry ``[l - 1, a]`` is
        ``C(f, a) * C(n - f, l - a) / C(n, l)`` (zero where impossible).

    """
    ones = np.arange(1, n + 1)[:, None]
    a = np.arange(f + 1)[None, :]
    hit = ones - a
    miss = n - f - hit
    valid = (hit >= 0) & (miss >= 0)
    hit_c = np.clip(hit, 0, n)
    miss_c = np.clip(miss, 0, n)
    log_p = (
        log_fact[f]
        - log_fact[a]
        - log_fact[f - a]
        + log_fact[n - f]
        - log_fact[hit_c]
        - log_fact[miss_c]
        - (log_fact[n] - log_fact[ones] - log_fact[n - ones])
    )
    return np.where(valid, np.exp(log_p), 0.0)


def single_child_rows(n, p, log_fact):
    """Return ``T[f, g]`` of one child for all buckets ``f = 0..n-1``.

    ``T[f, g] = P(max(f, f') = g)`` for one child with mutation rate ``p``,
    i.e. the elitist generation of the ``one_plus_one`` EA.  Every row is
    renormalised to sum to one, which redistributes the truncated binomial tail
    (see :func:`flip_count_distribution`).

    Parameters
    ----------
    n : int
        Number of bits.
    p : float
        Mutation rate, ``0 < p <= 1``.
    log_fact : numpy.ndarray
        ``log(k!)`` for ``k = 0..n``, see :func:`log_factorials`.

    Returns:
    -------
    numpy.ndarray
        Array of shape ``(n + 1, n + 1)``; row ``n`` is the terminal self-loop.

    """
    flip_pmf = flip_count_distribution(n, p, log_fact)
    ones = np.arange(1, n + 1)[:, None]
    table = np.zeros((n + 1, n + 1))
    for f in range(n):
        a = np.arange(f + 1)[None, :]
        weights = flip_pmf[:, None] * hypergeometric_matrix(n, f, log_fact)
        # f' = f + l - 2a (see the module docstring); elitist acceptance
        child = np.clip(f + ones - 2 * a, f, n)
        counts = np.bincount(
            (child - f).ravel(), weights=weights.ravel(), minlength=n - f + 1
        )
        table[f, f:] = counts / counts.sum()
    table[n, n] = 1.0
    return table


def transition_matrices(n, algorithm, portfolio):
    """Return the exact per-generation transition matrix of every action.

    Parameters
    ----------
    n : int
        Number of bits.
    algorithm : str
        ``one_plus_one`` or ``one_plus_lambda`` (see :func:`require_exact`).
    portfolio : sequence of int
        Action values: the mutation strength ``r`` for ``one_plus_one``, the
        population size ``lambda`` for ``one_plus_lambda``.

    Returns:
    -------
    numpy.ndarray
        Array of shape ``(len(portfolio), n + 1, n + 1)`` with
        ``T[i, f, g] = P(next bucket = g | bucket f, action i)``.

    Raises:
    ------
    ValueError
        If an action value is outside ``[1, n]`` (the environment rejects such
        actions, so they have no bucket-model meaning).

    """
    if any(value < 1 or value > n for value in portfolio):
        raise ValueError(
            f"Error: action values must be in [1, {n}], got {list(portfolio)}"
        )
    log_fact = log_factorials(n)
    single_cache = {}
    matrices = np.zeros((len(portfolio), n + 1, n + 1))
    for f in range(n):
        if algorithm == "one_plus_one":
            for index, value in enumerate(portfolio):
                p = value / n
                if p not in single_cache:
                    single_cache[p] = single_child_rows(n, p, log_fact)
                matrices[index, f, :] = single_cache[p][f, :]
            continue
        # one_plus_lambda: iid children with a fixed mutation rate 1/n
        p = 1.0 / n
        if p not in single_cache:
            single_cache[p] = single_child_rows(n, p, log_fact)
        cumulative = np.cumsum(single_cache[p][f, :])
        for index, value in enumerate(portfolio):
            row = np.zeros(n + 1)
            row[f] = cumulative[f] ** value
            row[f + 1 :] = cumulative[f + 1 :] ** value - cumulative[f:n] ** value
            matrices[index, f, :] = row
    matrices[:, n, n] = 1.0
    return matrices


def evaluations_per_generation(n, algorithm, portfolio):
    """Return the exact number of evaluations of one generation per action.

    ``one_plus_one`` uses exactly one evaluation per generation and
    ``one_plus_lambda`` exactly ``lambda`` (``mutate_independent`` evaluates every
    child, including the ones that are worse than the parent).

    Parameters
    ----------
    n : int
        Number of bits (unused; kept for symmetry with the other functions).
    algorithm : str
        ``one_plus_one`` or ``one_plus_lambda``.
    portfolio : sequence of int
        Action values.

    Returns:
    -------
    numpy.ndarray
        Array of length ``len(portfolio)`` with the cost ``c_a`` of one
        generation.

    """
    del n
    if algorithm == "one_plus_one":
        return np.ones(len(portfolio))
    return np.asarray(portfolio, dtype=float)


def backward_dp(n, matrices, costs):
    """Run the exact backward dynamic programme over the buckets.

    Parameters
    ----------
    n : int
        Number of bits.
    matrices : numpy.ndarray
        Transition matrices of shape ``(n_actions, n + 1, n + 1)``.
    costs : numpy.ndarray
        Evaluations of one generation per action.

    Returns:
    -------
    dict
        ``"values"``: ``V[f]``, the optimal expected evaluations to the optimum
        from bucket ``f``; ``"columns"``: ``V[f, a]``, the expected evaluations
        of "play ``a`` once, then the optimal policy" (the ``mean_evals``
        column); ``"fixed"``: ``W[a, f]``, the expected evaluations of always
        playing ``a``; ``"policy"``: ``argmin_a V[f, a]`` per bucket.  Entries
        are ``inf`` where the action cannot leave the bucket (``q_a(f) = 0``).

    """
    n_actions = matrices.shape[0]
    values = np.zeros(n + 1)
    columns = np.full((n_actions, n + 1), np.inf)
    fixed = np.zeros((n_actions, n + 1))
    for f in range(n - 1, -1, -1):
        self_loop = matrices[:, f, f]
        improve = 1.0 - self_loop
        feasible = improve > 0.0
        future = matrices[:, f, f + 1 :] @ values[f + 1 :]
        fixed_future = (matrices[:, f, f + 1 :] * fixed[:, f + 1 :]).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            columns[:, f] = np.where(feasible, (costs + future) / improve, np.inf)
            fixed[:, f] = np.where(feasible, (costs + fixed_future) / improve, np.inf)
        values[f] = columns[:, f].min()
    return {
        "values": values,
        "columns": columns,
        "fixed": fixed,
        "policy": np.argmin(columns, axis=0),
    }


def solve(n, algorithm, portfolio):
    """Compute the exact bucket model and its cost-to-go values for one combination.

    Parameters
    ----------
    n : int
        Number of bits.
    algorithm : str
        ``one_plus_one`` or ``one_plus_lambda``.
    portfolio : sequence of int
        Action values.

    Returns:
    -------
    dict
        ``"matrices"``, ``"costs"`` and the result of :func:`backward_dp`
        (``"values"``, ``"columns"``, ``"fixed"``, ``"policy"``).

    """
    require_exact("onemax", algorithm)
    matrices = transition_matrices(n, algorithm, portfolio)
    costs = evaluations_per_generation(n, algorithm, portfolio)
    result = backward_dp(n, matrices, costs)
    result["matrices"] = matrices
    result["costs"] = costs
    return result


def policy_table(n, algorithm, portfolio, solution=None):
    """Build the exact policy table of one combination.

    Parameters
    ----------
    n : int
        Number of bits.
    algorithm : str
        ``one_plus_one`` or ``one_plus_lambda``.
    portfolio : sequence of int
        Action values.
    solution : dict | None
        Result of :func:`solve`; computed when omitted.

    Returns:
    -------
    pandas.DataFrame
        One row per (bucket ``f_x = 0..n``, action) with the columns
        :data:`COLUMNS`, sorted by ``f_x`` and ``action``.  ``n_seeds = 0``,
        ``ci_low = ci_high = mean_evals`` and ``exact = 1``.

    """
    if solution is None:
        solution = solve(n, algorithm, portfolio)
    columns = solution["columns"]
    rows = []
    for f in range(n + 1):
        for index, value in enumerate(portfolio):
            mean = 0.0 if f == n else float(columns[index, f])
            rows.append(
                {
                    "f_x": f,
                    "action": index,
                    "choice": int(value),
                    "mean_evals": mean,
                    "n_seeds": 0,
                    "ci_low": mean,
                    "ci_high": mean,
                    "exact": 1,
                }
            )
    table = pd.DataFrame(rows, columns=list(COLUMNS))
    return table.sort_values(["f_x", "action"], kind="stable").reset_index(drop=True)


def fixed_action_table_rows(n, algorithm, portfolio, solution=None):
    """Build the fixed-action cost rows of one combination.

    Parameters
    ----------
    n : int
        Number of bits.
    algorithm : str
        ``one_plus_one`` or ``one_plus_lambda``.
    portfolio : sequence of int
        Action values.
    solution : dict | None
        Result of :func:`solve`; computed when omitted.

    Returns:
    -------
    list of dict
        One row per (start bucket ``0..n``, portfolio action) with
        ``expected_evals = W_a[start_f]`` and, for every start bucket, one row
        with ``action = -1, choice = 0`` holding the optimal ``V[start_f]``.

    """
    if solution is None:
        solution = solve(n, algorithm, portfolio)
    fixed = solution["fixed"]
    values = solution["values"]
    rows = []
    for f in range(n + 1):
        rows.append(
            {
                "n": n,
                "algorithm": algorithm,
                "start_f": f,
                "action": OPTIMAL_ACTION,
                "choice": 0,
                "expected_evals": 0.0 if f == n else float(values[f]),
            }
        )
        for index, value in enumerate(portfolio):
            rows.append(
                {
                    "n": n,
                    "algorithm": algorithm,
                    "start_f": f,
                    "action": index,
                    "choice": int(value),
                    "expected_evals": 0.0 if f == n else float(fixed[index, f]),
                }
            )
    return rows


def is_exact_table(path):
    """Return whether a CSV on disk is an exact table (``exact = 1``).

    ``main`` uses this to report when it replaces a simulation-derived table by
    an exact one; a table without an ``exact`` column is simulation-derived and
    returns ``False``.

    Parameters
    ----------
    path : str | Path
        CSV file to inspect.

    Returns:
    -------
    bool
        ``True`` if the file exists, has an ``exact`` column and at least one
        row with ``exact = 1``.

    """
    path = Path(path)
    if not path.is_file():
        return False
    header = pd.read_csv(path, nrows=1)
    if "exact" not in header.columns:
        return False
    marks = pd.read_csv(path, usecols=["exact"])["exact"]
    return bool((pd.to_numeric(marks, errors="coerce") == 1).any())


def _combination_list(problems, sizes, algorithms):
    """Return the requested (problem, n, algorithm) combinations, all exact."""
    for problem in problems:
        for algorithm in algorithms:
            require_exact(problem, algorithm)
    return [
        (problem, n, algorithm)
        for problem in problems
        for n in sizes
        for algorithm in algorithms
    ]


def parse_args(argv=None):
    """Parse command line arguments.

    Parameters
    ----------
    argv : list of str | None
        Arguments to parse, defaults to ``sys.argv[1:]``.

    Returns:
    -------
    argparse.Namespace
        Parsed arguments.

    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute the exact optimal bucket policies for the DACBench GA "
            "benchmark on OneMax (standalone script, not imported by the "
            "package). Refuses combinations for which the fitness bucket is "
            "not a sufficient statistic."
        ),
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=list(EXACT_PROBLEMS),
        choices=list(PROBLEMS),
        help="problem slugs (default: %(default)s); only 'onemax' is exact and "
        "every other choice is refused with an error",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_SIZES),
        help="problem sizes n (default: %(default)s)",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(EXACT_ALGORITHMS),
        choices=list(ALGORITHMS),
        help="algorithms (default: %(default)s); 'one_plus_lambda_lambda' is "
        "refused with an error",
    )
    parser.add_argument(
        "--portfolio",
        nargs="+",
        type=int,
        default=list(DEFAULT_PORTFOLIO),
        help="action values r / lambda (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        "--out-dir",
        type=Path,
        default=INSTANCE_DIR,
        dest="output_dir",
        help="output directory (default: the directory of this script)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute everything and report whether the files in the output "
        "directory are up to date instead of writing them",
    )
    args = parser.parse_args(argv)

    if any(size < 2 for size in args.sizes):
        parser.error("--sizes values must be >= 2")
    if any(value < 1 for value in args.portfolio):
        parser.error("--portfolio values must be >= 1")
    for size in args.sizes:
        if max(args.portfolio) > size:
            parser.error(
                f"--portfolio value {max(args.portfolio)} exceeds problem size "
                f"n={size}; the env only accepts action values in [1, n]"
            )
    for problem in args.problems:
        for algorithm in args.algorithms:
            reason = _not_exact_reason(problem, algorithm)
            if reason is not None:
                parser.error(
                    f"the bucket model is not exact for problem {problem!r} and "
                    f"algorithm {algorithm!r}: {reason}. Use "
                    f"derive_optimal_policy.py for a simulation-based table."
                )
    return args


def _frame_text(table):
    """Return the CSV text of a table as written by :meth:`pandas.DataFrame.to_csv`."""
    return table.to_csv(index=False)


def main(argv=None):
    """Regenerate the exact policy tables and the fixed-action cost table.

    Parameters
    ----------
    argv : list of str | None
        Command line arguments, defaults to ``sys.argv[1:]``.

    Returns:
    -------
    int
        Process exit code: 0 on success, 1 if ``--check`` found a stale file.

    """
    args = parse_args(argv)
    combinations = _combination_list(args.problems, args.sizes, args.algorithms)

    print(
        f"exact bucket DP: problems={args.problems} sizes={args.sizes} "
        f"algorithms={args.algorithms} portfolio={args.portfolio} "
        f"output_dir={args.output_dir} check={args.check}",
        flush=True,
    )
    start = time.perf_counter()
    stale = []
    summary = []
    fixed_rows = []
    for problem, n, algorithm in combinations:
        tic = time.perf_counter()
        solution = solve(n, algorithm, args.portfolio)
        table = policy_table(n, algorithm, args.portfolio, solution)
        fixed_rows.extend(
            fixed_action_table_rows(n, algorithm, args.portfolio, solution)
        )
        path = args.output_dir / output_name(problem, n, algorithm)
        elapsed = time.perf_counter() - tic

        text = _frame_text(table)
        if args.check:
            current = path.read_text() if path.is_file() else None
            up_to_date = current == text
            if not up_to_date:
                stale.append(path.name)
        else:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            if path.is_file() and not is_exact_table(path):
                print(
                    f"note: {path.name} is a simulation-derived table "
                    f"(no ``exact`` column or ``exact != 1``); replacing it by "
                    f"the exact one",
                    flush=True,
                )
            path.write_text(text)

        values = solution["values"]
        fixed = solution["fixed"]
        best = int(np.argmin(fixed[:, 0]))
        distinct = sorted(int(a) for a in np.unique(solution["policy"][:n]))
        summary.append(
            {
                "combination": f"{problem}_{n}_{algorithm}",
                "V0": float(values[0]),
                "best_fixed": best,
                "best_fixed_choice": int(args.portfolio[best]),
                "W0": float(fixed[best, 0]),
                "distinct": distinct,
                "seconds": elapsed,
                "file": path.name,
            }
        )
        print(
            f"[{problem:9s} n={n:4d} {algorithm:15s}] {elapsed:5.2f}s -> "
            f"{path.name}: V[0]={values[0]:.6f} evals, best fixed action "
            f"{args.portfolio[best]} needs {fixed[best, 0]:.6f}, "
            f"optimal policy uses {len(distinct)} distinct action(s)",
            flush=True,
        )

    fixed_table = pd.DataFrame(fixed_rows, columns=list(FIXED_COLUMNS))
    fixed_table = fixed_table.sort_values(
        ["n", "algorithm", "start_f", "action"], kind="stable"
    ).reset_index(drop=True)
    fixed_path = args.output_dir / FIXED_ACTIONS_FILE
    fixed_text = _frame_text(fixed_table)
    if args.check:
        current = fixed_path.read_text() if fixed_path.is_file() else None
        if current != fixed_text:
            stale.append(fixed_path.name)
    else:
        fixed_path.write_text(fixed_text)

    total = time.perf_counter() - start
    print(
        f"\nfixed-action costs: {len(fixed_table)} rows -> {fixed_path.name}",
        flush=True,
    )
    print(
        f"{'combination':38s} {'V[0]':>12s} {'W_a[0] (best fixed)':>20s} "
        f"{'V[0]/W_a[0]':>11s}  optimal policy (distinct actions)"
    )
    for entry in summary:
        print(
            f"{entry['combination']:38s} {entry['V0']:12.6f} "
            f"{entry['W0']:9.6f} (a={entry['best_fixed_choice']:<2d}) "
            f"{entry['V0'] / entry['W0']:11.4f}  {entry['distinct']}"
        )
    print(f"{'total':38s} {total:12.2f}s", flush=True)

    if args.check:
        if stale:
            print(
                f"\nout of date ({len(stale)} file(s)): {', '.join(stale)}; "
                f"rerun without --check to regenerate",
                flush=True,
            )
            return 1
        print("\nall files are up to date", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
