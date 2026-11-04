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
``ValueError`` for NaN. Non-finite discrete action indices follow the invalid-index
protocol in the training environment (episode ends, reward -1e8); the test environment
rejects them with a ``ValueError``.

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
   * - ``imp``
     - Δf − 0.5

``imp_minus_evals_scaling`` comes from the fork this benchmark is based on. All other
options are computed as in the Theory benchmark, which no longer offers
``imp_div_evals_new``: that name was accepted but never computed upstream, and PR #208
removed it in release 0.5.7, so the GA benchmark rejects it too.

Reward offset
~~~~~~~~~~~~~

``imp`` and ``imp_div_evals`` subtract 0.5 from the fitness improvement, both here and in
the Theory benchmark. The original LeadingOnesDAC implementation does not. The offset is
kept unchanged so that existing Theory results stay comparable; to convert to the upstream
values, add 0.5 to ``imp`` and 0.5/evals to ``imp_div_evals``.

Episodes
--------

An episode ends (``terminated``) when the optimum is found or the evaluation budget is used up.
It is ``truncated`` when the number of steps reaches ``cutoff``.
Steps return the gymnasium order ``(obs, reward, terminated, truncated, info)``.
The training environment uses a budget of 0.8·n² evaluations, like the Theory benchmark.
The test environment (``get_environment(test_env=True)``) uses ``cutoff`` instead (default 1e6);
if the config provides a ``test_set_path``, that environment is additionally switched to the
test instance set. ``gym.make("GA-v0")`` returns a *training* environment.

The environment attribute ``total_evals`` counts every evaluation of an episode, **including
the initial solution** (it is 1 directly after ``reset``), while a reward only counts the
evaluations of the step it belongs to. For example a step with λ = 4 adds 4 to
``total_evals``. The initial solution consumes one evaluation, so a test episode has
``cutoff − 1`` evaluations left to spend on steps: at most ``cutoff − 1`` steps of the
(1+1) EA, fewer steps of a population-based algorithm.

Like the Theory benchmark, ``GABenchmark`` has no ``get_benchmark(seed=...)`` method, so
``run_dacbench`` skips it in multi-benchmark sweeps; construct the benchmark directly in
that case.

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

Attribution
-----------

The environment is adapted from the MIT-licensed ``OLLGATheoryEnv`` in
`OneMax-DAC-Extension <https://github.com/taindp98/OneMax-DAC-Extension>`_.
Related work by the same group on RL-based control of the (1+(λ,λ)) GA on OneMax:
`arXiv:2502.20265 <https://arxiv.org/abs/2502.20265>`_,
`arXiv:2505.12982 <https://arxiv.org/abs/2505.12982>`_ and
`arXiv:2606.10129 <https://arxiv.org/abs/2606.10129>`_.
DACBench used to include a ``OneLLEnv``, which was removed in 2022 when the
Theory benchmark replaced it. This benchmark brings a GA benchmark back to DACBench.

References:

- B. Doerr, C. Doerr, F. Ebel. *From black-box complexity to designing new genetic algorithms.* Theoretical Computer Science, 2015.
- B. Doerr. *Probabilistic tools for the analysis of randomized optimization heuristics.*
  In *Theory of Evolutionary Computation*, Springer, 2020. arXiv:1801.06733.
- B. Doerr. *Better runtime guarantees via stochastic domination.* Theoretical Computer
  Science 773:115–137, 2019. arXiv:1801.04487.
- S. Böttcher, B. Doerr, F. Neumann. *Optimal fixed and adaptive mutation rates for the LeadingOnes problem.* PPSN 2010.
- D. Chen, M. Buzdalov, C. Doerr, N. Dang. *Using automated algorithm configuration for parameter control.* FOGA 2023. arXiv:2302.12334.
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
