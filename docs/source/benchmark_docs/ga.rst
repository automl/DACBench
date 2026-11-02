.. _ga:

=================
The GA Benchmark
=================

| **Task:** controlling the mutation strength or offspring population size of simple evolutionary algorithms on OneMax and LeadingOnes
| **Cost:** number of fitness evaluations until the optimum is found
| **Number of hyperparameters to control:** one integer (discrete portfolio by default, optionally a continuous value for the (1+1) EA that is truncated to an integer)
| **State Information:** user specified, highly flexible
| **Noise Level:** fairly large
| **Instance space:** OneMax and LeadingOnes of different sizes with different initial solutions

Like the :doc:`Theory benchmark <theory>`, this benchmark is cheap to run and is built on
well-studied evolutionary algorithms on pseudo-Boolean benchmark problems from runtime analysis.
In each step, the DAC controller picks one parameter of the algorithm: the mutation strength
of a (1+1) EA, or the offspring population size λ of a (1+λ) EA or a (1+(λ,λ)) GA.
The algorithm then runs one iteration with that parameter.
The goal is to reach the optimum with as few fitness evaluations as possible.

The benchmark is available as ``GABenchmark`` / ``GAEnv`` and is registered as ``"GA-v0"`` in
the gymnasium registry. It requires no extra dependencies.

Problems
--------

The ``problem`` option selects the objective function on bit strings of length n:

- ``OneMax`` (default): the number of one-bits; the all-ones string is optimal.
- ``LeadingOne``: the number of consecutive one-bits from the start of the string.

Algorithms
----------

The ``algorithm`` option selects the controlled algorithm. All three variants use elitist
acceptance: the offspring replaces the parent if its fitness is at least as good (≥).

.. list-table::
   :header-rows: 1
   :widths: 22 12 50 16

   * - ``algorithm``
     - Action
     - Iteration
     - Evaluations per step
   * - ``one_plus_one`` (default)
     - mutation strength r
     - (1+1) EA\ :sub:`>0`: each bit is flipped independently with probability r/n;
       draws that flip zero bits are resampled.
     - 1
   * - ``one_plus_lambda``
     - λ
     - (1+λ) EA: λ children are created independently, each by flipping
       ℓ ~ Bin\ :sub:`>0`\ (n, 1/n) bits. The best child is selected, with ties broken uniformly at random.
     - λ
   * - ``one_plus_lambda_lambda``
     - λ
     - (1+(λ,λ)) GA (Doerr, Doerr & Ebel 2015): mutation phase with rate p = λ/n
       (one ℓ ~ Bin\ :sub:`>0`\ (n, p) shared by all λ mutants, best mutant x' chosen with
       uniform tie-breaking), then λ biased crossovers with crossover bias c = 1/λ (each bit
       taken from x' with probability c, otherwise from the parent). x' is kept unless a
       crossover child is strictly better.
     - ≤ 2λ (see below)

For the (1+(λ,λ)) GA, evaluations are counted in a practice-aware way (Carvalho Pinto & Doerr 2018):
a crossover child that is identical to one of its parents is not evaluated and is not counted.
Hence a step uses at most 2λ evaluations. Set ``count_different_inds_only=False`` to count exactly 2λ evaluations per step.

Action Space
------------

By default, the action space is discrete. The action is an index into ``action_choices``
(default ``[1, 2, 4, 8, 16]``), i.e. the agent picks a value from a portfolio.

A continuous ``Box`` action space between ``min_action`` (default 1) and ``max_action``
(default: the largest n in the instance set) is only supported for ``one_plus_one``.
The value r is truncated to an integer (e.g. 2.7 becomes 2) before it is used.
Using a continuous action space with the λ-based algorithms raises a ``ValueError``.

Values outside of [1, n] and invalid action indices are treated differently in training and
test environments. The training environment ends the episode with a reward of -1e8.
The test environment clips the index to the portfolio and the value to [1, n].
Non-finite continuous values count as out of range: NaN and ±inf end the episode in the
training environment, while the test environment clips ±inf to [1, n] and raises a
``ValueError`` for NaN.

Observation Space
-----------------

The observation is configured with ``observation_description``, a comma-separated string
(default ``"n, f(x)"``, the same as for the Theory benchmark). Available entries:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Entry
     - Meaning
   * - ``n``
     - problem size
   * - ``f(x)``
     - fitness of the current solution
   * - ``delta_f(x)``
     - fitness change in the last step
   * - ``f(x)/n``
     - fitness normalised by the problem size
   * - ``r``
     - last action value, i.e. the mutation strength for ``one_plus_one`` (0 before the first step)
   * - ``lambda``
     - alias of ``r``, i.e. the last population size for the λ-based algorithms
   * - ``r_{t-k}``, ``lambda_{t-k}``, ``f(x)_{t-k}``
     - history of the respective value k steps ago, k = 0, …, 4

All entries have the observation-space domain [0, ∞), except ``f(x)/n``, which lies in [0, 1].

Reward
------

The reward is selected with ``reward_choice``. Below, Δf is the fitness improvement of
the step, evals is the number of evaluations it used and max_evals is the episode's
evaluation budget.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - ``reward_choice``
     - Reward
   * - ``imp_minus_evals`` (default)
     - Δf − evals
   * - ``imp_minus_evals_normalised``
     - (Δf − evals) / max_evals
   * - ``imp_minus_evals_scaling``
     - (Δf − evals) / n
   * - ``minus_evals``
     - −evals
   * - ``minus_evals_normalised``
     - −evals / max_evals
   * - ``imp_div_evals``
     - (Δf − 0.5) / evals
   * - ``imp_div_evals_new``
     - (Δf − 0.5) / evals (same formula; kept for parity with Theory)
   * - ``imp``
     - Δf − 0.5

``imp_minus_evals_scaling`` comes from the fork this benchmark is based on. All other
options are computed as in the Theory benchmark.

Reward offset
~~~~~~~~~~~~~

``imp`` and ``imp_div_evals`` subtract 0.5 from the fitness improvement, both here and in
the Theory benchmark. The original LeadingOnesDAC implementation does not. The offset is
kept unchanged so that existing Theory results stay comparable; when comparing against the
upstream code, add 0.5 to the values of those two rewards.

Episodes
--------

An episode ends (``terminated``) when the optimum is found or the evaluation budget is used up.
It is ``truncated`` when the number of steps reaches ``cutoff``.
Steps return the gymnasium order ``(obs, reward, terminated, truncated, info)``.
The training environment uses a budget of 0.8·n² evaluations, like the Theory benchmark.
The test environment (``get_environment(test_env=True)``) uses ``cutoff`` instead (default 1e6).

Instance Sets
-------------

Instance sets are stored in ``dacbench/instance_sets/ga/`` as CSV files with the columns
``id,size,initObj``. ``size`` is the problem size n. ``initObj`` is either ``random``
(uniformly random initial solution) or an integer. For OneMax, the integer is the number
of one-bits, placed at random positions. For LeadingOnes, it is the number of leading ones.

- ``onemax_{20,50,100,200}.csv``: initial solutions ``random``, 0, ⌊n/4⌋, n/2 and n−1.
- ``onemax_{20,50,100}_easy.csv``: start at n−1.
- ``onemax_{20,50,100}_random.csv``: uniformly random initial solutions.
- ``onemax_{300,500,1000,2000}.csv``: sizes mirroring the prior work, with initial solutions
  ``random`` and n/2 (the prior work starts from n/2 one-bits).
- ``leadingones_{50,100,200}.csv``: same initial-solution ladder as the Theory benchmark's ``lo_rls_*`` sets.

The sets can be regenerated with the ``sample_ga_instances.ipynb`` notebook.

.. _ga_reference_policies:

Reference Policies (simulation-derived)
----------------------------------------

Unlike the Theory benchmark (RLS on LeadingOnes, where the optimal policy is known and
computable), this benchmark has **no theorem-derived optimal policy**: no closed-form
optimal control policy is known for these GA variants on OneMax and LeadingOnes. What
ships instead is a **simulation-derived reference policy** (``best_static``, with tables
for 9 of the 18 combinations) and two baseline rules (``success_rule`` and
``random_policy``).

The reference policies are **approximate reference policies, not optima**. They solve a
simulated model of the problem, they carry Monte-Carlo error, and for LeadingOnes the
model is known to be incomplete. They answer one question: "is a learned policy clearly
better than the obvious fixed action?".

Method: policy iteration over on-policy state pools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``dacbench/instance_sets/ga/derive_optimal_policy.py`` (a standalone script that the
package never imports) derives, for each (problem, n, algorithm) and each fitness
bucket, the expected cost of every action in the portfolio (default
``[1, 2, 4, 8, 16]``). The derivation is a **policy iteration over on-policy state
pools**:

0. *Fresh-state round.* For every bucket ``f(x) = b < n`` and every action, one
   generation is simulated with the operators of this very environment, starting
   from freshly drawn states with fitness ``b``. All actions of a bucket share the
   same random stream (**common random numbers**), so that the comparison of actions
   is paired and much less noisy. A **backward dynamic programme** from ``f(x) = n``
   (cost 0) down to ``f(x) = 0`` gives the initial value function and policy:
   ``V(b, a) = E[e] / q + E[V(b') | b' > b]`` with ``q = P(b' > b)``. Here
   ``V(b, a)`` is the expected number of evaluations to the optimum if action ``a``
   is played in bucket ``b`` and the derived policy is followed afterwards; the
   derived action of a bucket is ``argmin_a V(b, a)``. The division by ``q`` resolves
   the self-loops: a generation that does not improve stays in the bucket, and its
   evaluation cost has to be paid ``1/q`` times on average.
1. *On-policy state pools.* Full episodes are run in the real environment (test mode,
   cutoff ``10 n^2``, the shipped instance ladder) under the current policy. For every
   bucket, the state at which an episode *enters* the bucket is recorded. Fitness is
   monotone, so an episode enters each bucket at most once; repeated generations in a
   bucket do not change the fitness, and recording all of them would weight states by
   their waiting time instead of by the entry distribution the value function is
   defined on. The pools therefore contain the actual bitstrings, and in particular
   the actual suffixes, that the policy encounters.
2. *One-step lookahead from the pool.* For every bucket and action, states are drawn
   from the bucket's pool with replacement, one generation is simulated and the
   evaluations and the successor bucket are recorded. Again all actions share the
   same state and random stream per sample. The new transition estimates are combined
   by the same backward dynamic programme, which yields the next value function and
   the next greedy policy.
3. *Iteration and convergence.* Steps 1 and 2 are repeated with the new policy until
   the policy stops changing or ``--rounds`` on-policy rounds are used. Pools are built
   with fixed RNG streams, so identical invocations produce bit-identical CSVs, and a
   fallback bucket sees the same fresh states in every round. The script prints, per
   combination, whether the policy **converged** before ``--rounds`` was exhausted,
   together with the per-round policy-change counts, and repeats the convergence status
   in its summary. A table of a non-converged combination is a snapshot of the last
   round, not a fixed point of the iteration.
4. *Sampling effort.* Sampling stops for an action once it observed
   ``MIN_IMPROVEMENTS`` improvements, because the statistical error of ``V(b, a)`` is
   dominated by the error of ``q(b, a)``. Actions that are hopelessly worse than the
   best action of their bucket stop early, and the total number of samples per bucket
   and action is capped.
5. *Confidence intervals.* ``ci_low`` and ``ci_high`` are 95 % bootstrap percentile
   intervals over the per-sample costs, computed from a fixed RNG seed. A resample
   without a single improvement has infinite cost, so a row with very few improvements
   reports an infinite upper bound.

**Convergence.** With the default ``--rounds 3`` none of the nine shipped tables
reaches a fixed point of the policy iteration: in the last round the policy still
changes in 2 to 34 of the buckets (2, 2, 6, 8, 10, 17, 19, 24 and 34 for the nine
combinations). The shipped tables are therefore **snapshots of the last round, not
converged solutions of the bucket model**. They are reproducible — an identical
invocation produces a bit-identical CSV — but they should not be read as "the optimal
policy of the simulation". A larger ``--rounds`` iterates further and yields a
different table, which is exactly the kind of change the 1.05 tolerance is there to
catch.

Buckets with fewer than five pooled states (for example buckets that the instance
ladder barely reaches) fall back to fresh-state sampling for that bucket; the number of
fallback buckets is reported. Fresh states are still the conditional initial
distribution of the benchmark (for LeadingOnes, ``initObj=b`` is exactly a uniform
initial string conditioned by fitness ``b``), so this is a sound fallback.

A row whose action never improved in the simulated samples of its bucket has
``mean_evals = inf`` (and ``ci_low = ci_high = inf``): **no simulated sample of that
action improved**, so ``q = 0`` in the sample and no finite estimate exists. It does
*not* mean that the action cannot improve, nor that it failed within an evaluation
budget. Such rows are never selected.

The result is one file per shipped combination,
``optimal_policy_{problem}_{n}_{algorithm}.csv`` with ``problem`` in
``{onemax, leadingones}``, ``n`` in ``{50, 100, 200}`` and ``algorithm`` in
``{one_plus_one, one_plus_lambda, one_plus_lambda_lambda}``, with columns::

    f_x,action,choice,mean_evals,n_seeds,ci_low,ci_high

One row per (bucket ``f_x``, portfolio index ``action``), sorted by ``f_x`` and then
``action``. ``choice`` is the parameter value (``r`` or λ), ``mean_evals`` the
simulated cost-to-go ``V(f(x), action)``, ``n_seeds`` the number of simulated
transitions of that row and ``ci_low``/``ci_high`` the bootstrap interval. The dynamic
programme is exact for the bucket model, but its inputs are Monte-Carlo estimates: this
is why the files contain ``n_seeds`` and confidence intervals.

Why is the fresh-state round not enough? For OneMax it is exact: the problem is
symmetric, so states with the same ``f(x)`` are exchangeable and the successor fitness
of a uniform mutation depends only on ``f(x)``. For LeadingOnes it is **not**: fitness
``b`` fixes the prefix ``0..b-1``, while the suffix ``b+1..n-1`` is whatever the run
arrived with. In a real run that suffix is correlated with the history, it typically
contains far more ones than a freshly drawn uniform suffix, and it is exactly what the
crossover of the (1+(λ,λ)) GA exploits. A fresh-suffix model therefore misestimates
the value of the crossover.

Validation: which of the 18 combinations ship
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shipping a table is a claim that the table is a useful reference. A combination
therefore ships only if its derived policy is **at most 1.05 times the evaluations of
the best fixed portfolio action on both validation protocols A and B, and never
significantly worse** than it. Both protocols are full-episode comparisons in test mode
(cutoff ``10 n^2``, the shipped instance ladder, episodes run until the optimum or the
cutoff) and both are **paired**: every policy runs in a fresh environment and meets the
same seeds in the same order, so each policy sees exactly the same (seed, instance)
pairs.

- **A**: 12 seeds (0–11) — the selection protocol that chose the shipped tables.
- **B**: 12 held-out seeds (1000–1011), disjoint from A.

The selection was made on A; the held-out B was not used to choose anything and is the
check that actually counts.

.. list-table:: Derived policy vs. best fixed action, ratio of evaluations on the two paired protocols (A: seeds 0–11, B: held-out seeds 1000–1011). A table ships only if the ratio is at most 1.05 on **both** protocols and is never significantly worse; "–" = not measured.
   :header-rows: 1
   :widths: 12 5 20 8 8 17 22

   * - Problem
     - n
     - Algorithm
     - A
     - B
     - best fixed action
     - status
   * - OneMax
     - 50
     - (1+1) EA
     - 0.882
     - 0.801
     - r = 1
     - shipped
   * - OneMax
     - 50
     - (1+λ) EA
     - 0.968
     - 0.997
     - λ = 2
     - shipped
   * - OneMax
     - 50
     - (1+(λ,λ)) GA
     - 1.047
     - 1.088
     - λ = 1
     - excluded: failed validation
   * - OneMax
     - 100
     - (1+1) EA
     - 0.948
     - 1.007
     - r = 1
     - shipped
   * - OneMax
     - 100
     - (1+λ) EA
     - 1.041
     - 1.088
     - λ = 1
     - excluded: failed validation
   * - OneMax
     - 100
     - (1+(λ,λ)) GA
     - 0.837
     - 0.791
     - λ = 1
     - shipped
   * - OneMax
     - 200
     - (1+1) EA
     - 0.923
     - 0.859
     - r = 1
     - shipped
   * - OneMax
     - 200
     - (1+λ) EA
     - 0.847
     - 0.909
     - λ = 1
     - shipped
   * - OneMax
     - 200
     - (1+(λ,λ)) GA
     - 0.879
     - 0.990
     - λ = 8
     - shipped
   * - LeadingOne
     - 50
     - (1+1) EA
     - 1.095
     - –
     - r = 1
     - excluded: structural
   * - LeadingOne
     - 50
     - (1+λ) EA
     - 1.136
     - –
     - λ = 2
     - excluded: structural
   * - LeadingOne
     - 50
     - (1+(λ,λ)) GA
     - 1.083
     - –
     - λ = 1
     - excluded: structural
   * - LeadingOne
     - 100
     - (1+1) EA
     - 0.935
     - 0.965
     - r = 1
     - shipped
   * - LeadingOne
     - 100
     - (1+λ) EA
     - 0.953
     - 1.357
     - λ = 2
     - excluded: failed validation
   * - LeadingOne
     - 100
     - (1+(λ,λ)) GA
     - 0.990
     - 0.963
     - λ = 1
     - shipped
   * - LeadingOne
     - 200
     - (1+1) EA
     - 1.016
     - 1.062
     - r = 1
     - excluded: failed validation
   * - LeadingOne
     - 200
     - (1+λ) EA
     - 1.073
     - –
     - λ = 1
     - excluded: structural
   * - LeadingOne
     - 200
     - (1+(λ,λ)) GA
     - 1.269
     - 1.316
     - λ = 1
     - excluded: structural

``best fixed action`` is the portfolio value that achieved the smallest mean number of
evaluations in protocol A; it is the reference a user should use for a combination
without a table. Which action is best is itself noisy: on the held-out seeds a different
portfolio value wins in several combinations (OneMax n=50 (1+λ): λ = 2 on A, λ = 4 on
B; OneMax n=100 (1+(λ,λ)): λ = 1 on A, λ = 4 on B; OneMax n=200 (1+λ): λ = 1 on A,
λ = 2 on B), usually by a few percent. The tolerance is
therefore applied to the ratio, not to the identity of the winner. Nine combinations
ship, and their ratios lie between 0.79 and 1.01 (0.84–0.99 on A, 0.79–1.01 on B).

**Excluded because the validation failed** — the derivation ran, the derived policy was
not better than the best fixed action:

- OneMax n=50 (1+(λ,λ)) GA: A 1.047, B 1.088 versus fixed λ = 1. Note that 1.047 is
  *below* the 1.05 tolerance on the selection protocol and still **worse** than the
  fixed action: the tolerance is an upper bound that has to hold, not a licence to be
  worse. This table ships in no version.
- OneMax n=100 (1+λ) EA: A 1.041, B 1.088 versus fixed λ = 1 — the same pattern.
- LeadingOne n=100 (1+λ) EA: A 0.953, B 1.357 versus fixed λ = 2 — fine on the
  selection seeds, clearly worse on the held-out seeds.
- LeadingOne n=200 (1+1) EA: A 1.016, B 1.062 versus fixed r = 1 — worse on both, and
  above the tolerance on the held-out seeds.

**Excluded as structurally unvalidated** — the fitness-only bucket model cannot
represent the problem, so no validation run can change the verdict. Only the A ratio
was measured for most of them, B where it was available:

- LeadingOne n=50 with the (1+1) EA (1.095), the (1+λ) EA (1.136) and the
  (1+(λ,λ)) GA (1.083);
- LeadingOne n=200 (1+λ) EA (1.073);
- LeadingOne n=200 (1+(λ,λ)) GA (1.269 / 1.316; the paired 95 % confidence interval of
  the difference excludes 0, the clearest failure).

Why the LeadingOne combinations fail is structural, not statistical:

- OneMax is symmetric (``f(x) = sum_i x_i``): two states with the same ``f(x)`` are
  exchangeable, so ``f(x)`` is a sufficient statistic and the bucket table is exact
  for the bucket model, whatever the history. This is why a OneMax table can ship even
  where the margin is thin.
- LeadingOne is **not** symmetric: ``f(x) = b`` fixes the prefix ``0..b-1`` and leaves
  the suffix ``b+1..n-1`` free, and that suffix decides where the next leading one
  comes from, both for a mutation and, even more, for the crossover of the (1+(λ,λ))
  GA. The on-policy pools see the suffixes of the *current* policy's trajectory, which
  is where the iteration becomes circular: buckets that the current policy reaches late
  are polled with states that a better policy would not produce. Repeating the
  lookahead with more on-policy rounds (an H-step lookahead) does not recover the
  ordering; the measured ratios stay where the table shows them. The tables also keep
  recommending λ = 2 in buckets where λ = 1 is measurably better.
- That LeadingOne happens to pass only at n=100 (and only there for two of the three
  algorithms) is an empirical result of these measurements, not something the bucket
  model predicts.

**The honest conclusion.** The reference policies that ship are better than the best
fixed action, but only modestly: 0.79–1.01 times its evaluations (the best case is
OneMax n=100 with the (1+(λ,λ)) GA at 0.791 on held-out seeds; OneMax n=100 with the
(1+1) EA is at parity there, 1.007, and 0.948 on the selection seeds).
They are **approximate reference policies, not optima**: a policy that needs 0.85 times
the evaluations of a constant action is a useful baseline for a DAC method, not a target
that has to be matched, and the tables are snapshots of a policy iteration that has not
converged (see above). The derivation is exact for the bucket model of OneMax by
symmetry, but the bucket model cannot capture the LeadingOne suffix, which is why nine
of the 18 combinations have no table at all.

What this means for users:

- All 18 combinations remain fully usable for training and evaluation; the environment
  itself does not depend on the tables.
- For the nine combinations without a table the reference is the **measured best fixed
  action** of that combination (see the table above) or ``success_rule``.
  ``best_static`` raises a ``ValueError`` for them that names the category, the measured
  ratios and the best fixed action.
- ``success_rule`` is not a good substitute for the (1+(λ,λ)) GA on LeadingOne n=200
  either: in the same 12 episodes it needs 29 903 evaluations, where the fixed action
  λ = 1 needs 9 648. Use the fixed action there.

Regenerating the tables::

    python dacbench/instance_sets/ga/derive_optimal_policy.py

This derives exactly the 9 shipped combinations. ``--all`` (or
``--include-unvalidated``) derives the remaining nine as well; they are written for
research use and are not shipped. Useful flags are ``--rounds`` (on-policy rounds after
the fresh-state round, default 3; ``0`` reproduces the fresh-state method),
``--episodes-per-round`` (episodes per round used for the state pools, default 40),
``--n-seeds`` (batch size of the simulated transitions), ``--problems``, ``--sizes``,
``--algorithms``, ``--portfolio``, ``--seed`` and ``--out-dir``.

Baseline policies
~~~~~~~~~~~~~~~~~

``dacbench/envs/env_utils/ga_baselines.py`` provides three callables with the
``policy(env, state) -> int`` signature used by
:class:`~dacbench.agents.generic_agent.GenericAgent`:

- ``best_static``: looks up ``f(x)`` in the derived table and returns
  ``argmin_a mean_evals``. Unknown buckets (gaps, the optimum) resolve to the nearest
  bucket; ``inf`` rows are never selected.
- ``success_rule``: the one-fifth success rule. For ``one_plus_one`` the mutation
  strength follows ``r <- r*F`` after an improvement and ``r <- r*F^(-1/4)``
  otherwise (Rechenberg 1973; multiplicative form of Kern et al. 2004, Eq. 3; the
  factor ``F = 1.5`` and the exponent are the ones of Auger, GECCO'09, Table 1). For
  the λ-based algorithms λ is updated in the opposite direction,
  ``lambda <- lambda/F`` on an improvement and ``lambda <- lambda*F^(1/4)``
  otherwise, because a larger population increases the per-iteration success
  probability; this is the self-adjusting (1+(λ,λ)) GA of Doerr & Doerr (GECCO 2015;
  Algorithmica 2018). For the (1+λ) EA the same λ-update is used as an *adaptation*:
  the published analysis covers the (1+(λ,λ)) GA only.
- ``random_policy``: a uniform random action, the trivial baseline.

The stateful success rule tracks only the previous fitness and resets at the start of
an episode; ``best_static`` is stateless and ``random_policy`` keeps only its own random
generator. The comparison table is generated with::

    python examples/run_ga_baselines.py
    python examples/run_ga_baselines.py --quick

It evaluates ``best_static``, ``success_rule`` and ``random_policy`` on the OneMax and
LeadingOnes instance sets for n = 50, 100, 200, all three algorithms and several seeds,
and prints the mean number of evaluations to the optimum with a 95% confidence
interval.

Theory reference policies
~~~~~~~~~~~~~~~~~~~~~~~~~

For the (1+(λ,λ)) GA on OneMax the literature does provide analytic policies, which may
not be representable exactly with a discrete action portfolio:

- the fitness-dependent choice λ = √(n / (n − f(x))), known to yield linear expected
  runtime (Doerr, Doerr & Ebel 2015);
- the self-adjusting 1/5-th success rule (Doerr & Doerr 2018), which is the rule that
  ``success_rule`` implements for the λ-based algorithms.

No analogue is known for the (1+λ) EA or for LeadingOnes; the simulation-derived tables
above are the reference there.

Attribution
-----------

The environment is adapted from the MIT-licensed ``OLLGATheoryEnv`` in
`OneMax-DAC-Extension <https://github.com/taindp98/OneMax-DAC-Extension>`_.
Related work by Tai Nguyen, Phong Le, André Biedenkapp, Carola Doerr and Nguyen Dang
on RL-based control of the (1+(λ,λ)) GA on OneMax:
`arXiv:2502.20265 <https://arxiv.org/abs/2502.20265>`_,
`arXiv:2505.12982 <https://arxiv.org/abs/2505.12982>`_ and
`arXiv:2606.10129 <https://arxiv.org/abs/2606.10129>`_.
DACBench used to include a ``OneLLEnv``, which was removed in 2022 when the
Theory benchmark replaced it. This benchmark brings a GA benchmark back to DACBench.

References:

- B. Doerr, C. Doerr, F. Ebel. *From black-box complexity to designing new genetic algorithms.* Theoretical Computer Science, 2015.
- B. Doerr, C. Doerr. *Optimal parameter choices through self-adjustment: applying the 1/5-th rule in discrete settings.* GECCO 2015. arXiv:1504.03212.
- B. Doerr, C. Doerr. *Optimal static and self-adjusting parameter choices for the (1+(λ,λ)) genetic algorithm.* Algorithmica, 2018. DOI 10.1007/s00453-017-0354-9.
- I. Rechenberg. *Evolutionsstrategie: Optimierung technischer Systeme nach Prinzipien der biologischen Evolution.* Frommann-Holzboog, 1973.
- S. Kern, S. D. Müller, N. Hansen, D. Büche, J. Očenášek, P. Koumoutsakos. *Learning probability distributions in continuous evolutionary algorithms – a comparative review.* Natural Computing 3:77–112, 2004. DOI 10.1023/B:NACO.0000023416.59689.4e.
- A. Auger. *Benchmarking the (1+1) evolution strategy with one-fifth success rule on the BBOB-2009 function testbed.* GECCO 2009 companion.
- E. Carvalho Pinto, C. Doerr. *Towards a more practice-aware runtime analysis of evolutionary algorithms.* arXiv:1812.00493, 2018.
- A. Biedenkapp, N. Dang, M. S. Krejca, F. Hutter, C. Doerr. *Theory-inspired parameter control benchmarks for dynamic algorithm configuration.* GECCO 2022. arXiv:2202.03259.


.. automodule:: dacbench.benchmarks.ga_benchmark
    :members:
    :show-inheritance:

.. automodule:: dacbench.envs.ga
    :members:
    :show-inheritance:
