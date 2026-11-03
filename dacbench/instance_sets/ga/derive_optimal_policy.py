"""Derive simulation-based reference policies for the GA benchmark.

This is a *standalone* script.  It is never imported by the ``dacbench`` package
and no part of the package depends on it: the CSVs it produces are reference
data, not code.  Run it from a checkout, e.g.::

    python dacbench/instance_sets/ga/derive_optimal_policy.py

With no arguments it regenerates exactly the simulation-derived shipped files
(``optimal_policy_{problem}_{n}_{algorithm}.csv``) for the default scope:
OneMax and LeadingOne, n in {50, 100, 200}, all three algorithms, portfolio
[1, 2, 4, 8, 16].  Six of the ten shipped tables are **not** produced here: for
OneMax with the (1+1) EA and with the (1+lambda) EA the fitness bucket is a
sufficient statistic, so ``exact_optimal_policy.py`` computes the *exact*
optimal table instead (``exact = 1``).  The default scope of this script is
therefore exactly the four combinations that still need simulation.  It never
overwrites an exact table: if the output file already carries ``exact = 1`` the
combination is skipped with a message, and only ``--force`` overrides that.
``--all`` additionally derives the eight combinations without a validated table;
they are written for research, but they are **not** shipped, because no
validated reference policy exists for them.  A cheap exploratory run looks
like::

    python dacbench/instance_sets/ga/derive_optimal_policy.py \
        --problems onemax --sizes 100 --algorithms one_plus_lambda_lambda \
        --n-seeds 20 --rounds 1 --episodes-per-round 10

The result is an **approximate reference policy, not an optimum** (see
the section on the shipped subset at the end of this docstring).  With the
default ``--rounds 3`` none of the shipped tables reaches a fixed point of the
policy iteration; they are snapshots of the last round.

Why is the reference policy derived from simulation?
----------------------------------------------------
The files hold, for every fitness bucket ``f(x)`` and every portfolio action,
the expected *evaluations* needed to reach the optimum when that action is used
once and the derived policy is followed afterwards.  They are deliberately
obtained by simulating the *same operators the benchmark uses*
(``(1+1) EA_{>0}``, ``(1+lambda) EA``, ``(1+(lambda,lambda)) GA`` with elitist
acceptance) instead of from closed-form theorems: no analytic runtime formula
enters the derivation, so the derived policy cannot silently disagree with the
implementation (mutation strength resampled until > 0, practice-aware evaluation
counting, tie-breaking, ...).  The only modelling assumption is the fitness
bucket abstraction described below.

Fitness buckets and why a fresh-state DP is not enough
------------------------------------------------------
The DP state is the fitness value ``f(x) in {0, ..., n}``.

* OneMax (``f(x) = sum_i x_i``) is symmetric in the bit positions: for a
  uniformly chosen set of flipped positions the successor fitness depends only
  on ``f(x)``.  Drawing new states with a fresh, uniformly random arrangement of
  the bits for every sample is therefore exact.  This is why the first round of
  the derivation (below) is still the classical fresh-state DP for OneMax.
* LeadingOne (``f(x)`` = number of leading ones) is *not* symmetric.  Fitness
  ``b`` fixes the prefix ``0..b-1`` to one and bit ``b`` to zero, but the suffix
  ``b+1..n-1`` carries information: in a real run it is whatever the run arrived
  with, it is correlated with the history, and it usually contains far more ones
  than a freshly drawn uniform suffix.  That suffix is exactly what the
  crossover of the ``(1+(lambda,lambda)) GA`` exploits, and what decides where
  the next LeadingOne improvement comes from.  Drawing a fresh uniform suffix
  for every sample therefore misestimates the successor distribution.  Measured
  in full episodes (``test_env=True``, cutoff ``10*n*n``, 6 seeds) the fresh
  state table was *worse than a constant action* for the ``(1+(lambda,lambda))``
  GA on LeadingOne: ~978 vs. 750 evaluations at n=50 and ~11470 vs. ~9379 at
  n=200 (always using the first portfolio action).  A reference policy that is
  worse than a constant action is not acceptable, which motivates the
  policy-iteration method below.

Method: policy iteration over on-policy state pools
--------------------------------------------------
For a fixed (problem, n, algorithm) the script does:

0. **Round 0 -- fresh-state DP.**  For every bucket and action, one generation
   is simulated from freshly drawn states with ``f(x) = b`` (the code path of
   the original derivation).  A backward dynamic programme from ``f(x) = n``
   (cost 0) yields the initial value estimate ``V_0`` and policy
   ``pi_0(b) = argmin_a V_0(b, a)`` with::

       V(b) = min_a V(b, a)
       V(b, a) = E[e(b, a)] / q(b, a) + E[V(b') | b' > b]

   ``e`` is the evaluations of the generation, ``q(b, a) = P(b' > b)`` and the
   division by ``q`` resolves the self-loops (generations without an
   improvement).
1. **On-policy state pools.**  Episodes are run in the real benchmark
   environment (``GAEnv`` through ``GABenchmark``, test mode, cutoff
   ``10*n*n``, the shipped instance ladder) under the current policy ``pi``,
   with ``--episodes-per-round`` different seeds and hence different initial
   solutions.  For every bucket the state at which an episode *enters* the
   bucket is recorded.  Fitness is monotone, so an episode enters each bucket
   at most once; repeated generations inside a bucket do not change the state,
   and recording all of them would weight states by their waiting time (the
   states the policy is stuck in) instead of by the entry distribution the
   value function is defined on.  Per bucket at most ``POOL_CAP`` entry states
   are kept (deterministic reservoir sampling).  The resulting pools describe
   the *actual* bitstrings, and hence suffixes, that the policy encounters.
2. **One-step lookahead from the pool.**  For every bucket and action, states
   are sampled from that bucket's pool with replacement, one generation of the
   action is simulated and its evaluations and successor bucket are recorded.
   All actions share the same starting state and the same random stream per
   sample (**common random numbers**), exactly as in round 0, so actions are
   compared on paired samples.  Buckets with fewer than ``MIN_POOL_STATES``
   pooled states fall back to fresh-state sampling for that bucket; the number
   of fallback buckets is reported.  Fresh states are the conditional initial
   distribution of the benchmark (for LeadingOne, ``initObj=b`` is exactly a
   uniform initial string conditioned on fitness ``b``), so they are a sound
   fallback for buckets that the ladder barely reaches.
3. **Backward DP and policy improvement.**  The new transition estimates are
   combined by the same backward DP from ``f(x) = n`` down to ``f(x) = 0``,
   yielding ``V_{k}`` and ``pi_{k+1} = argmin_a V_k(b, a)``.  The policy-change
   count between ``pi_k`` and ``pi_{k+1}`` is reported.  Steps 1-3 are repeated
   with ``pi_{k+1}`` until the policy stops changing for all buckets or
   ``--rounds`` on-policy rounds have been used.  (``--rounds 0`` stops after
   the fresh-state DP.)  Whether the policy converged before ``--rounds`` was
   exhausted is printed per combination and repeated in the summary: a table
   of a non-converged combination is a snapshot of the last round, not a fixed
   point of the policy iteration.
4. **Sampling and confidence intervals.**  Sampling stops for an action once it
   observed ``MIN_IMPROVEMENTS`` improvements, because the statistical error of
   ``V(b, a)`` is dominated by the error of ``q(b, a)``.  Actions that are
   already more than ``HOPELESS_FACTOR`` times worse than the best action of the
   bucket stop early; an action without a single improvement is dropped once
   the best action of the bucket has ``3 * HOPELESS_FACTOR`` improvements,
   because then the 95% upper bound ``3 / N`` of its improvement probability is
   more than ``HOPELESS_FACTOR`` times smaller.  Dropped actions cannot become
   the derived choice when the remaining samples are added, and their rows are
   reported with the samples they happen to have.  The total number of samples
   per bucket and action is capped at
   ``MAX_SAMPLES_FACTOR[algorithm] * n_seeds``.
5. An action that never improved in the simulated samples gets
   ``mean_evals = inf`` (it cannot be shown to make progress), and all three
   numeric columns are ``inf`` for that row.  ``q(b, a) > 0`` holds for every
   valid action in the true process; ``inf`` is therefore a statement about the
   simulated sample, not about the algorithm.  If *no* action of a bucket
   improves, that bucket (and every bucket below it) is unreachable under the
   derived policy and the script raises an error instead of writing an
   inconsistent table; increase ``--n-seeds`` (or ``--episodes-per-round``).
6. 95% confidence intervals are bootstrap percentile intervals over the
   per-sample costs, using a fixed RNG seed, so the output is reproducible: the
   identical invocation produces bit-identical CSVs.  Resamples of a row without
   a single improvement have infinite cost, so a row whose samples contain very
   few improvements reports an infinite upper bound.

Shipped subset
--------------
A table ships only if its policy needs at most 1.05 times the evaluations of the
best fixed portfolio action on **both** paired seed protocols (A: seeds 0-11,
the selection protocol; B: held-out seeds 1000-1011, the out-of-sample check)
and is never significantly worse there.  The eight excluded combinations, their
category and their measured ratios are in
:data:`dacbench.envs.env_utils.ga_baselines.UNVALIDATED_COMBINATIONS`;
``docs/source/benchmark_docs/ga.rst`` holds the measurements and the reasoning.

Output
------
One CSV per (problem, n, algorithm),
``optimal_policy_{problem}_{n}_{algorithm}.csv`` with the problem slug in lower
case (``onemax``, ``leadingones``), the columns::

    f_x,action,choice,mean_evals,n_seeds,ci_low,ci_high

``f_x`` is the fitness bucket, ``action`` the portfolio index, ``choice`` the
action value (``r`` for ``one_plus_one``, ``lambda`` otherwise), ``mean_evals``
the expected cost-to-go ``V(b, a)`` under the derived policy, ``n_seeds`` the
number of simulated transitions of that row, and ``ci_low``/``ci_high`` the 95%
CI of ``mean_evals``.  Rows are sorted by ``f_x`` and ``action``.

The six exact tables of ``exact_optimal_policy.py`` use the same file names,
the same column order and the same meaning of ``mean_evals``, plus a trailing
``exact`` column (``exact = 1``, ``n_seeds = 0``, ``ci_low = ci_high =
mean_evals``).  This script never writes that column and never overwrites a file
that has it (see ``--force``).

The terminal bucket ``f(x) = n`` is included so that the table covers the whole
fitness range of "n, f(x)" observations: no further evaluations are needed
there, whichever action is played, so all its rows have ``mean_evals = 0``,
``ci_low = ci_high = 0`` and ``n_seeds = 0``.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from dacbench.abstract_benchmark import objdict
from dacbench.benchmarks import GABenchmark
from dacbench.envs.env_utils.ga_baselines import (
    EXACT_COMBINATIONS,
    SHIPPED_COMBINATIONS,
    UNVALIDATED_COMBINATIONS,
    format_ratios,
    is_exact_table,
    policy_file_name,
)
from dacbench.envs.ga import ALGORITHMS, GAEnv
from dacbench.envs.theory import LeadingOne, OneMax

PROBLEMS = {"onemax": ("OneMax", OneMax), "leadingones": ("LeadingOne", LeadingOne)}

#: Directory with the shipped instance sets.  It must be absolute:
#: ``GABenchmark`` looks up a same-named CSV in the current working directory
#: first, so a relative name would silently change behaviour.
INSTANCE_DIR = Path(__file__).resolve().parent

DEFAULT_SIZES = (50, 100, 200)
DEFAULT_PORTFOLIO = (1, 2, 4, 8, 16)
DEFAULT_N_SEEDS = 200
DEFAULT_SEED = 0

#: The (problem, n, algorithm) combinations whose table ships with the package,
#: and the subset of them with an *exact* table, are re-exported from
#: :mod:`dacbench.envs.env_utils.ga_baselines` (:data:`SHIPPED_COMBINATIONS`,
#: :data:`EXACT_COMBINATIONS`) so that this script, ``exact_optimal_policy.py``
#: and the package cannot drift apart.

#: The default scope of this script: the shipped combinations that still need
#: simulation, i.e. the shipped ones without an exact table.  ``main`` skips an
#: exact output file even where it is requested explicitly, unless ``--force``
#: is given.
SIMULATED_COMBINATIONS = tuple(
    combination
    for combination in SHIPPED_COMBINATIONS
    if combination not in EXACT_COMBINATIONS
)

#: Combinations without a validated table.  The data is the single source of
#: truth of :data:`dacbench.envs.env_utils.ga_baselines.UNVALIDATED_COMBINATIONS`
#: (re-exported here under the historical name); every entry gives the category,
#: the measured ratios of the derived policy's evaluations to those of the best
#: fixed action (A: seeds 0-11, the selection protocol; B: held-out seeds
#: 1000-1011, ``None`` where not measured), the best fixed portfolio value and a
#: one-line reason.  They are only derived with ``--all`` and are not shipped.
EXCLUDED_COMBINATIONS = UNVALIDATED_COMBINATIONS

#: number of on-policy policy-improvement rounds *after* the fresh-state round 0
DEFAULT_ROUNDS = 3
#: number of full episodes per (problem, n, algorithm) and round, used to build
#: the on-policy state pools
DEFAULT_EPISODES_PER_ROUND = 40
#: evaluation budget of those episodes in units of n**2 (test mode)
EPISODE_CUTOFF_FACTOR = 10
#: buckets with fewer pooled states than this fall back to fresh-state sampling
MIN_POOL_STATES = 5
#: maximum number of states kept per bucket (deterministic reservoir sampling)
POOL_CAP = 200

#: number of observed improvements after which an action of a bucket is
#: considered well estimated (the error of V(b, a) is dominated by the error of
#: the improvement probability)
MIN_IMPROVEMENTS = 12
#: an action that is this much worse than the best action of its bucket stops
#: being sampled
HOPELESS_FACTOR = 4.0
#: maximum number of samples per bucket and action in units of ``n_seeds``
MAX_SAMPLES_FACTOR = {
    "one_plus_one": 25,
    "one_plus_lambda": 5,
    "one_plus_lambda_lambda": 5,
}
#: distance between the random streams of two consecutive samples
STREAM_STRIDE = 2**64
#: number of bootstrap resamples for the confidence intervals
N_BOOTSTRAP = 200
#: confidence level of the reported intervals
CI_LEVEL = 0.95

COLUMNS = ("f_x", "action", "choice", "mean_evals", "n_seeds", "ci_low", "ci_high")

# tags that keep the random streams of different combinations apart
_PROBLEM_TAGS = {slug: i for i, slug in enumerate(PROBLEMS)}
_ALGORITHM_TAGS = {alg: i for i, alg in enumerate(ALGORITHMS)}


# ``output_name`` and ``_is_exact_table`` keep their historical names here and
# delegate to ``ga_baselines``, which owns the naming scheme and the exact marker.
def output_name(problem, n, algorithm):
    """Return the CSV file name for one (problem, n, algorithm) combination."""
    return policy_file_name(problem, n, algorithm)


_is_exact_table = is_exact_table


def make_generation_env(problem_name, algorithm, portfolio, n=None):
    """Build a GAEnv that is used as a one-generation simulator.

    The env is never reset and no episode is run: the state and the random
    generator are assigned per simulated transition, so that ``_generation`` can
    be called directly.  This reuses the env's operator composition (including
    elitist acceptance and practice-aware evaluation counting) instead of
    re-implementing it.  ``n`` is the problem size the operators work on; the
    placeholder instance of the config is never read, and ``n`` may be omitted
    only by callers that set ``env.n`` themselves.
    """
    config = objdict(
        {
            "name": "optimal_policy_derivation",
            "algorithm": algorithm,
            "problem": problem_name,
            "discrete_action": True,
            "action_choices": list(portfolio),
            "observation_description": "n, f(x)",
            "reward_choice": "minus_evals",
            "count_different_inds_only": True,
            "seed": 0,
            # requirements of AbstractEnv, never used by the derivation
            "instance_set": {0: objdict({"size": 2, "initObj": 0})},
            "benchmark_info": {},
            "cutoff": 1,
            "reward_range": (-np.inf, np.inf),
            "observation_space": gym.spaces.Box(
                low=0.0, high=np.inf, shape=(2,), dtype=np.float64
            ),
            "action_space": gym.spaces.Discrete(len(portfolio)),
        }
    )
    env = GAEnv(config, test_env=False)
    if n is not None:
        env.n = n
    return env


def make_episode_env(problem, n, algorithm, portfolio):
    """Build a real benchmark environment used to collect on-policy states.

    The environment is a ``GABenchmark`` test environment on the shipped
    instance ladder (``onemax_{n}.csv`` / ``leadingones_{n}.csv``), with the
    evaluation budget ``EPISODE_CUTOFF_FACTOR * n**2``.  Consecutive
    ``reset`` calls progress through the ladder round-robin, so different
    episodes see different initial solutions.
    """
    config = {
        "problem": PROBLEMS[problem][0],
        "algorithm": algorithm,
        "action_choices": list(portfolio),
        "instance_set_path": str(INSTANCE_DIR / f"{problem}_{n}.csv"),
        "cutoff": EPISODE_CUTOFF_FACTOR * n * n,
    }
    return GABenchmark(config=config).get_environment(test_env=True)


def _estimate_bucket_means(evals, succ, bucket, values):
    """Estimate ``V(b, a) = E[e] / q + E[V(b') | b' > b]`` for all actions.

    Returns ``V(b, a)`` (``inf`` for actions without a single improvement) and
    the number of observed improvements, both per action.
    """
    n_actions = len(evals)
    means = np.full(n_actions, np.inf)
    counts = np.zeros(n_actions, dtype=int)
    for a in range(n_actions):
        if not evals[a]:
            continue
        succ_a = np.asarray(succ[a])
        improved = succ_a > bucket
        counts[a] = int(improved.sum())
        if counts[a] == 0:
            # no improvement observed: no finite estimate
            continue
        # the non-improving samples have b' == bucket, whose value is unknown
        # here; dividing the evaluation cost by q resolves that self-loop
        means[a] = (
            np.asarray(evals[a]).sum() / counts[a] + values[succ_a[improved]].mean()
        )
    return means, counts


def bootstrap_cost(evals, succ, bucket, values, rng, n_boot=N_BOOTSTRAP):
    """Bootstrap the expected cost-to-go of one action in one bucket.

    Resamples the simulated transitions with replacement and recomputes
    ``E[e] / q + E[V(b') | b' > b]`` for each resample.  Resamples without a
    single improvement have infinite cost.
    """
    n_samples = evals.size
    idx = rng.integers(0, n_samples, size=(n_boot, n_samples))
    counts = (succ[idx] > bucket).sum(axis=1)
    costs = np.full(n_boot, np.inf)
    ok = counts > 0
    if ok.any():
        evaluated = evals[idx][ok]
        improved = succ[idx][ok]
        # samples with b' == bucket contribute no future cost: their self-loop
        # is already resolved by dividing the evaluation cost by the number of
        # improvements
        future = np.where(improved > bucket, values[improved], 0.0)
        costs[ok] = evaluated.sum(axis=1) / counts[ok] + future.sum(axis=1) / counts[ok]
    return costs


def summarize_bootstrap(costs, level=CI_LEVEL):
    """Return percentile (low, high) bounds of a bootstrap distribution.

    Both bounds are ``inf`` if every resample was infeasible, the upper bound is
    ``inf`` if any resample was infeasible.
    """
    finite = costs[np.isfinite(costs)]
    if finite.size == 0:
        return np.inf, np.inf
    alpha = 100.0 * (1.0 - level) / 2.0
    low = float(np.percentile(finite, alpha))
    if finite.size < costs.size:
        return low, np.inf
    return low, float(np.percentile(finite, 100 - alpha))


def solve_bucket(
    env,
    problem_cls,
    n,
    bucket,
    portfolio,
    values,
    n_seeds,
    seed,
    tags,
    max_samples,
    state_pool=None,
    min_improvements=MIN_IMPROVEMENTS,
    hopeless_factor=HOPELESS_FACTOR,
):
    """Simulate transitions out of one bucket and estimate all action values.

    Samples are drawn in batches of ``n_seeds``.  Sample ``s`` starts from the
    same state and uses the same random stream for every action (common random
    numbers), so actions are compared on paired samples.  An action stops being
    sampled once it observed ``min_improvements`` improvements or once it can no
    longer become the derived choice (it is ``hopeless_factor`` times worse than
    the best action of the bucket, or it improved at all in far fewer samples
    than that action), so the rows of the CSV may have different ``n_seeds``.
    Buckets close to the optimum need the most samples: there, a single
    generation improves with a tiny probability, and every extra improvement
    reduces the statistical error of ``V(b, a)``.  ``values`` holds the cost-to-go
    of all buckets ``> bucket``; ``state_pool`` holds on-policy states with
    ``f(x) == bucket``, which are drawn with replacement instead of generating
    fresh ones.  Returns ``V(b, a)``, the 95% CI bounds and the number of samples
    per action, plus a dict with the sample and improvement counts.
    """
    n_actions = len(portfolio)
    evals = [[] for _ in range(n_actions)]
    succ = [[] for _ in range(n_actions)]
    active = np.ones(n_actions, dtype=bool)
    states = []
    op_states = []
    # one master stream for the states and one for the operations of this
    # bucket; the operations of sample s are the segment of the master stream
    # starting at s * STREAM_STRIDE, restored for every action so that all
    # actions see the same random numbers
    state_master = np.random.default_rng((seed, *tags, bucket, 0))
    op_master = np.random.default_rng((seed, *tags, bucket, 1))
    used = 0
    means = np.full(n_actions, np.inf)
    counts = np.zeros(n_actions, dtype=int)
    use_pool = state_pool is not None and len(state_pool) > 0

    while True:
        # states for the new samples.  With a pool, draw with replacement from
        # the states the current policy actually visits in this bucket (their
        # suffixes matter for LeadingOne); otherwise draw fresh states.  In both
        # cases sample s is shared by all actions.
        if use_pool:
            indices = state_master.integers(0, len(state_pool), size=n_seeds)
            draws = [copy.deepcopy(state_pool[int(i)]) for i in indices]
        else:
            draws = [
                problem_cls(n=n, rng=state_master, initObj=bucket)
                for _ in range(n_seeds)
            ]
        for state in draws:
            states.append(state)
            op_states.append(copy.deepcopy(op_master.bit_generator.state))
            op_master.bit_generator.advance(STREAM_STRIDE)
        for a in range(n_actions):
            if not active[a]:
                continue
            for s in range(used, used + n_seeds):
                op_master.bit_generator.state = op_states[s]
                env.x = states[s]
                env.np_random = op_master
                evals[a].append(env._generation(int(portfolio[a])))
                succ[a].append(env.x.fitness)
        used += n_seeds

        means, counts = _estimate_bucket_means(evals, succ, bucket, values)
        done = np.array(
            [not evals[a] or counts[a] >= min_improvements for a in range(n_actions)]
        )
        if (done | ~active).all() or used >= max_samples:
            break
        best = int(np.argmin(means))
        if np.isfinite(means[best]):
            # Actions that are far worse than the best one cannot become the
            # derived choice: stop wasting transitions on them.  An action
            # without a single improvement has no finite estimate; 0 events in
            # ``N`` samples bound its improvement probability by roughly ``3/N``
            # (95%), so it is at least ``3 / best_count`` times worse than the
            # best action of the bucket.  It is dropped once that ratio exceeds
            # ``hopeless_factor``.
            hopeless = np.where(
                counts > 0,
                means > hopeless_factor * means[best],
                counts[best] >= 3.0 * hopeless_factor,
            )
            active &= ~hopeless

    ci_low = np.full(n_actions, np.inf)
    ci_high = np.full(n_actions, np.inf)
    for a in range(n_actions):
        if counts[a] == 0:
            continue
        boot_rng = np.random.default_rng((seed, *tags, bucket, a, 2))
        costs = bootstrap_cost(
            np.asarray(evals[a], dtype=np.float64),
            np.asarray(succ[a]),
            bucket,
            values,
            boot_rng,
        )
        ci_low[a], ci_high[a] = summarize_bootstrap(costs)

    info = {
        "n_samples": [len(col) for col in evals],
        "n_improvements": counts.tolist(),
        "pool_size": len(state_pool) if use_pool else 0,
    }
    return means, ci_low, ci_high, info


def collect_policy_state_pools(
    problem, n, algorithm, portfolio, policy, episodes, seed, tags
):
    """Collect the states a policy visits in full benchmark episodes.

    Runs ``episodes`` test-mode episodes of the real environment (shipped
    instance ladder, evaluation budget ``EPISODE_CUTOFF_FACTOR * n**2``) with
    the action ``policy[f(x)]`` and records, per fitness bucket, the state at
    which an episode enters that bucket.  Fitness is monotone, so an episode
    enters each bucket at most once; repeated generations inside a bucket do not
    change the state, so recording them would only weight states by their
    waiting time and bias the estimates towards the states the policy is stuck
    in.  Per bucket at most ``POOL_CAP`` entry states are kept with
    deterministic reservoir sampling.  ``policy`` maps a fitness bucket to the
    action index of the previous round.  Returns the pools (one per bucket
    ``0..n-1``) and the number of episodes that entered each bucket, before
    reservoir sampling.
    """
    env = make_episode_env(problem, n, algorithm, portfolio)
    pools = [[] for _ in range(n)]
    seen = [0] * n
    pool_rng = np.random.default_rng((seed, *tags, 3))

    for episode in range(episodes):
        env.reset(seed=int(seed) * 100003 + episode)
        # Fitness never decreases, so an episode enters each bucket at most
        # once.  Only the state at that entry is recorded: later generations in
        # the same bucket do not change the state, so recording every visit
        # would only weight states by their waiting time (the states the policy
        # is stuck in) and bias the transition estimates.
        entered = set()
        while True:
            bucket = int(env.x.fitness)
            if 0 <= bucket < n and bucket not in entered:
                entered.add(bucket)
                seen[bucket] += 1
                if len(pools[bucket]) < POOL_CAP:
                    pools[bucket].append(copy.deepcopy(env.x))
                else:
                    slot = int(pool_rng.integers(seen[bucket]))
                    if slot < POOL_CAP:
                        pools[bucket][slot] = copy.deepcopy(env.x)
            action = int(policy.get(bucket, 0))
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    return pools, seen


def run_policy_pass(
    env,
    problem_cls,
    n,
    algorithm,
    portfolio,
    n_seeds,
    seed,
    tags,
    pools=None,
):
    """Estimate one policy's value function and the next greedy policy.

    Runs the backward DP over the buckets.  Every bucket uses its on-policy
    state pool when it has at least ``MIN_POOL_STATES`` states and fresh states
    otherwise; ``pools`` is ``None`` for the fresh-state round 0.  Returns the
    cost-to-go ``V`` per bucket (``V[n] = 0``), the greedy policy
    ``{bucket: action index}``, the CSV rows of this pass and statistics
    (fallback buckets, total samples).
    """
    max_samples = MAX_SAMPLES_FACTOR[algorithm] * n_seeds
    values = np.full(n + 1, np.inf)
    values[n] = 0.0
    rows = []
    best_actions = {}
    n_fallback = 0
    n_samples_total = 0

    for bucket in range(n - 1, -1, -1):
        state_pool = None
        if pools is not None:
            if len(pools[bucket]) >= MIN_POOL_STATES:
                state_pool = pools[bucket]
            else:
                n_fallback += 1
        means, ci_low, ci_high, info = solve_bucket(
            env,
            problem_cls,
            n,
            bucket,
            portfolio,
            values,
            n_seeds,
            seed,
            tags,
            max_samples,
            state_pool=state_pool,
        )
        best = int(np.argmin(means))
        if not np.isfinite(means[best]):
            raise RuntimeError(
                f"bucket f(x)={bucket} of {n} {algorithm} is infeasible: no "
                "action improved in "
                f"{max(info['n_samples'])} simulated transitions "
                f"(pool size {info['pool_size']}). Increase --n-seeds or "
                "--episodes-per-round."
            )
        values[bucket] = means[best]
        best_actions[bucket] = best
        n_samples_total += sum(info["n_samples"])

        for a in range(len(portfolio)):
            rows.append(
                {
                    "f_x": bucket,
                    "action": a,
                    "choice": int(portfolio[a]),
                    "mean_evals": float(means[a]),
                    "n_seeds": int(info["n_samples"][a]),
                    "ci_low": float(ci_low[a]),
                    "ci_high": float(ci_high[a]),
                }
            )

    # the terminal bucket needs no further evaluations, whatever action is
    # played; it is listed so that the table covers the whole range f(x) = 0..n
    for a in range(len(portfolio)):
        rows.append(
            {
                "f_x": n,
                "action": a,
                "choice": int(portfolio[a]),
                "mean_evals": 0.0,
                "n_seeds": 0,
                "ci_low": 0.0,
                "ci_high": 0.0,
            }
        )
    stats = {
        "n_fallback": n_fallback,
        "n_samples_total": n_samples_total,
        "max_samples": max_samples,
    }
    return values, best_actions, rows, stats


def derive_optimal_policy(
    problem,
    n,
    algorithm,
    portfolio,
    n_seeds,
    seed,
    rounds=DEFAULT_ROUNDS,
    episodes_per_round=DEFAULT_EPISODES_PER_ROUND,
):
    """Derive the optimal policy table for one (problem, n, algorithm).

    Runs the fresh-state round 0 and up to ``rounds`` on-policy policy
    improvement rounds (see the module docstring).  Returns the table (one row
    per (bucket, action) with the columns ``COLUMNS``) and per-combination
    statistics (``inf`` rows, best actions, convergence, per-round changes,
    fallback buckets, samples).
    """
    problem_name, problem_cls = PROBLEMS[problem]
    size_tag = DEFAULT_SIZES.index(n) if n in DEFAULT_SIZES else 1000 + n
    env = make_generation_env(problem_name, algorithm, portfolio, n)

    best_actions = None
    rows = None
    round_stats = []
    # The random streams do not depend on the round: a fallback bucket therefore
    # sees exactly the same fresh states in every round (only the successor
    # values change), and once the policy is stable the pass is bit-identical,
    # so the loop can detect convergence instead of chasing Monte-Carlo noise.
    tags = (_PROBLEM_TAGS[problem], size_tag, _ALGORITHM_TAGS[algorithm])
    for round_index in range(rounds + 1):
        pools = None
        pool_sizes = None
        if round_index > 0:
            pools, _seen = collect_policy_state_pools(
                problem,
                n,
                algorithm,
                portfolio,
                best_actions,
                episodes_per_round,
                seed,
                tags,
            )
            pool_sizes = [len(pool) for pool in pools]

        _values, new_actions, rows, pass_stats = run_policy_pass(
            env,
            problem_cls,
            n,
            algorithm,
            portfolio,
            n_seeds,
            seed,
            tags,
            pools,
        )
        if best_actions is None:
            changes = None
        else:
            changes = sum(int(new_actions[b] != best_actions[b]) for b in range(n))
        round_stats.append(
            {
                "round": round_index,
                "changes": changes,
                "fallback": pass_stats["n_fallback"],
                "samples": pass_stats["n_samples_total"],
                "pool_states": None if pool_sizes is None else sum(pool_sizes),
                "pool_min": None if pool_sizes is None else min(pool_sizes),
                "pool_max": None if pool_sizes is None else max(pool_sizes),
            }
        )
        best_actions = new_actions
        if changes == 0:
            break

    # ``changes`` is None in the fresh-state round and non-zero in every round
    # that still improved the policy; the loop above breaks on the first round
    # without a change, so a finite loop end with the last ``changes`` equal to
    # 0 means that the policy converged before ``--rounds`` was exhausted.
    converged = round_stats[-1]["changes"] == 0

    table = pd.DataFrame(rows, columns=list(COLUMNS))
    table = table.sort_values(["f_x", "action"], kind="stable")
    table = table.reset_index(drop=True)
    stats = {
        "n_inf_rows": int(np.isinf(table["mean_evals"]).sum()),
        "n_distinct_best": len(set(best_actions.values())),
        "n_samples_total": int(table["n_seeds"].sum()),
        "converged": converged,
        "n_rounds": len(round_stats) - 1,
        "best_at": {
            b: int(best_actions[b]) for b in (0, n // 2, n - 1) if b in best_actions
        },
        "rounds": round_stats,
    }
    return table, stats


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Derive simulation-based reference policies for the DACBench GA "
            "benchmark (standalone script, not imported by the package)."
        ),
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=list(PROBLEMS),
        choices=list(PROBLEMS),
        help="problem slugs (default: %(default)s)",
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
        default=list(ALGORITHMS),
        choices=list(ALGORITHMS),
        help="algorithms (default: %(default)s)",
    )
    parser.add_argument(
        "--portfolio",
        nargs="+",
        type=int,
        default=list(DEFAULT_PORTFOLIO),
        help="action values r / lambda (default: %(default)s)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=DEFAULT_N_SEEDS,
        help="batch size of simulated transitions per bucket and action "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help="number of on-policy policy-improvement rounds after the "
        "fresh-state round 0; 0 regenerates the fresh-state table only "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--episodes-per-round",
        type=int,
        default=DEFAULT_EPISODES_PER_ROUND,
        help="number of full episodes per on-policy round used to collect the "
        "state pools (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="base RNG seed (default: %(default)s)",
    )
    parser.add_argument(
        "--all",
        "--include-unvalidated",
        action="store_true",
        dest="include_unvalidated",
        help="also derive the eight combinations without a validated table "
        "(OneMax n=50 (1+(lambda,lambda)) GA, LeadingOne n=50 with all three "
        "algorithms, LeadingOne n=100 (1+lambda) EA, LeadingOne n=200 with all "
        "three algorithms); they are not shipped and best_static refuses them. "
        "Exact tables are still skipped unless --force is given",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an output file even if it is an exact table "
        "(exact = 1). Use with care: it replaces the exact optimum by a "
        "simulation estimate",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="output directory (default: the directory of this script)",
    )
    args = parser.parse_args(argv)

    if args.n_seeds < 1:
        parser.error("--n-seeds must be >= 1")
    if args.rounds < 0:
        parser.error("--rounds must be >= 0")
    if args.episodes_per_round < 1:
        parser.error("--episodes-per-round must be >= 1")
    if any(value < 1 for value in args.portfolio):
        parser.error("--portfolio values must be >= 1")
    for n in args.sizes:
        if max(args.portfolio) > n:
            parser.error(
                f"--portfolio value {max(args.portfolio)} exceeds problem size "
                f"n={n}; the env only accepts action values in [1, n]"
            )
    return args


def main(argv=None):
    """Regenerate the simulation-derived optimal policy CSVs.

    Combinations whose output file is already an exact table (``exact = 1``) are
    skipped with a message unless ``--force`` is given, and the default scope is
    :data:`SIMULATED_COMBINATIONS`, i.e. the shipped combinations that still need
    simulation.
    """
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    requested = [
        (problem, n, algorithm)
        for problem in args.problems
        for n in args.sizes
        for algorithm in args.algorithms
    ]
    excluded = [c for c in requested if c in EXCLUDED_COMBINATIONS]
    if args.include_unvalidated:
        selected = requested
    else:
        selected = [c for c in requested if c in SHIPPED_COMBINATIONS]
    combinations = []
    exact_skipped = []
    exact_on_disk = []
    for combination in selected:
        if not args.force and combination in EXACT_COMBINATIONS:
            exact_skipped.append(output_name(*combination))
            continue
        path = args.out_dir / output_name(*combination)
        if not args.force and _is_exact_table(path):
            exact_on_disk.append(path.name)
            continue
        combinations.append(combination)
    if exact_skipped:
        print(
            f"skipping the {len(exact_skipped)} exact combination(s) of this "
            "selection: exact_optimal_policy.py computes their tables exactly "
            f"({', '.join(exact_skipped)}); pass --force to replace an exact "
            "table by a simulation estimate anyway",
            flush=True,
        )
    if exact_on_disk:
        print(
            "skipping files that already carry exact = 1: "
            f"{', '.join(exact_on_disk)}; pass --force to overwrite them anyway",
            flush=True,
        )

    print(
        f"deriving optimal policies: problems={args.problems} sizes={args.sizes} "
        f"algorithms={args.algorithms} portfolio={args.portfolio} "
        f"n_seeds={args.n_seeds} rounds={args.rounds} "
        f"episodes_per_round={args.episodes_per_round} seed={args.seed} "
        f"all={args.include_unvalidated}",
        flush=True,
    )
    if not combinations:
        if exact_skipped or exact_on_disk:
            print(
                "nothing to do: every selected combination already has an "
                "exact table or its output file is one; pass --force to "
                "replace one with a simulation estimate",
                flush=True,
            )
        else:
            print(
                "no simulated combination selected: the requested combinations "
                "either have an exact table (see exact_optimal_policy.py) or no "
                "validated table at all (pass --all to derive the latter anyway)",
                flush=True,
            )
    start = time.perf_counter()
    summary = []
    for problem, n, algorithm in combinations:
        tic = time.perf_counter()
        table, stats = derive_optimal_policy(
            problem,
            n,
            algorithm,
            args.portfolio,
            args.n_seeds,
            args.seed,
            rounds=args.rounds,
            episodes_per_round=args.episodes_per_round,
        )
        path = args.out_dir / output_name(problem, n, algorithm)
        table.to_csv(path, index=False)
        elapsed = time.perf_counter() - tic
        summary.append(
            {
                "combination": f"{problem}_{n}_{algorithm}",
                "seconds": elapsed,
                "inf_rows": stats["n_inf_rows"],
                "distinct_best": stats["n_distinct_best"],
                "samples": stats["n_samples_total"],
                "best_at": stats["best_at"],
                "rounds": stats["rounds"],
                "converged": stats["converged"],
                "n_rounds": stats["n_rounds"],
            }
        )
        rounds_text = " ".join(
            "fresh"
            if entry["changes"] is None
            else f"r{entry['round']}:{entry['changes']}chg/{entry['fallback']}fb"
            for entry in stats["rounds"]
        )
        status = "converged" if stats["converged"] else "NOT converged"
        print(
            f"[{problem:11s} n={n:4d} {algorithm:24s}] "
            f"{elapsed:7.1f}s -> {path.name} "
            f"({stats['n_inf_rows']} inf rows, "
            f"{stats['n_distinct_best']} distinct best actions, "
            f"{status} after {stats['n_rounds']} on-policy round(s), "
            f"{rounds_text})",
            flush=True,
        )

    total = time.perf_counter() - start
    print("\nsummary", flush=True)
    print(
        f"{'combination':42s} {'seconds':>8s} {'samples':>9s} {'inf':>6s} "
        f"{'best':>5s} {'conv':>5s}  {'rounds (changes/fallback)':32s} "
        "best action at f(x)"
    )
    for entry in summary:
        rounds_text = " ".join(
            "fresh"
            if e["changes"] is None
            else f"r{e['round']}:{e['changes']}/{e['fallback']}"
            for e in entry["rounds"]
        )
        print(
            f"{entry['combination']:42s} {entry['seconds']:8.1f} "
            f"{entry['samples']:9d} {entry['inf_rows']:6d} "
            f"{entry['distinct_best']:5d} {str(entry['converged'])[0]:>5s}  "
            f"{rounds_text:32s} {entry['best_at']}"
        )
    print(f"{'total':42s} {total:8.1f}", flush=True)
    if summary:
        n_converged = sum(entry["converged"] for entry in summary)
        print(
            f"\nconvergence: {n_converged} of {len(summary)} combination(s) "
            f"reached a stable policy before --rounds={args.rounds} was "
            f"exhausted (reproducibility comes from the fixed RNG streams; a "
            f"table of a non-converged combination is a snapshot of the last "
            f"round, not a fixed point of the policy iteration)",
            flush=True,
        )

    if excluded:
        width = max(len(f"{p}_{s}_{a}") for p, s, a in excluded)
        print(
            f"\n{'combination':{width}s} {'category':18s} "
            f"{'derived/best-fixed':25s} reason"
        )
        for problem, n, algorithm in excluded:
            entry = EXCLUDED_COMBINATIONS[(problem, n, algorithm)]
            print(
                f"  {f'{problem}_{n}_{algorithm}':{width}s} {entry['category']:18s} "
                f"{format_ratios(entry['ratios']):25s} {entry['reason']}"
            )
        print("  (ratios: A = seeds 0-11 (selection), B = held-out seeds 1000-1011)")
        if args.include_unvalidated:
            print(
                "\nthe combinations above are derived for research use only and "
                "are NOT shipped: best_static raises for them"
            )
        else:
            print(
                "\nthe combinations above have no validated reference policy "
                "(see the module docstring); they were skipped"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
