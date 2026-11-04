"""Theory-derived baseline policies for the GA benchmark.

Three table-free baselines for :mod:`dacbench.envs.ga`, which controls the
mutation strength ``r`` of the (1+1) EA or the population size ``lambda`` of the
(1+lambda) EA and the (1+(lambda,lambda)) GA:

- ``theory_policy``: the theory-derived parameter choice of a combination.
- ``success_rule``: the one-fifth success rule, on ``r`` or on ``lambda``.
- ``random_policy``: a uniform random action.

These are baselines, not optima. Where a theorem exists the value is the published
one (``r = 1`` for the (1+1) EA on OneMax, the fitness-dependent
``lambda = round(sqrt(n / max(1, n - f(x))))`` of Doerr, Doerr & Ebel for the
(1+(lambda,lambda)) GA on OneMax); otherwise it is the smallest admissible value.
**No reference policy tables ship with this benchmark**, so a learned policy is
compared against the best fixed action measured in the run, as the cited prior
work does (Biedenkapp, Dang, Krejca, Hutter & Doerr, GECCO 2022,
arXiv:2202.03259).

Policy API
----------
Every policy is a callable ``policy(env, state) -> int`` returning an action
**index** in ``[0, len(env.action_choices))``, as
:class:`dacbench.agents.generic_agent.GenericAgent` calls it; ``state`` is the
observation and is unused here. ``policy.reset()`` restarts the policy (a no-op for
``theory_policy``). With a continuous action space (``one_plus_one`` only) there is
no portfolio: ``theory_policy`` and ``success_rule`` return the integer ``r``,
``random_policy`` samples it from ``[1, n]``.

References:
----------
- One-fifth success rule for a step size: Rechenberg (1973); multiplicative form of
  Kern et al., Natural Computing 3:77-112, 2004, Eq. 3 (also Auger, GECCO 2009,
  Table 1).
- One-fifth success rule for ``lambda`` of the (1+(lambda,lambda)) GA: Doerr &
  Doerr, GECCO 2015, arXiv:1504.03212; Algorithmica 80(5):1658-1709, 2018.
- Fitness-dependent ``lambda`` of the (1+(lambda,lambda)) GA on OneMax: Doerr,
  Doerr & Ebel, Theoretical Computer Science 567:87-104, 2015; used as the
  ``theory`` baseline by Chen, Buzdalov, Doerr & Dang, FOGA 2023, arXiv:2302.12334.
- Mutation rate of the (1+1) EA on OneMax: Doerr, "Probabilistic tools for the
  analysis of randomized optimization heuristics", Springer 2020, arXiv:1801.06733.
- Mutation rate of the (1+1) EA on LeadingOnes: Boettcher, Doerr & Neumann, PPSN
  2010 (``p ~ 1.59/n``); Doerr, "Better runtime guarantees via stochastic
  domination", Theoretical Computer Science 773:115-137, 2019, arXiv:1801.04487.
"""

from __future__ import annotations

from math import sqrt

import numpy as np

#: Maps the benchmark's problem names to their lower-case file-name stems.
PROBLEM_FILE_STEMS = {"OneMax": "onemax", "LeadingOne": "leadingones"}

#: Algorithms for which theory policies and success rules exist.
ALGORITHMS = ("one_plus_one", "one_plus_lambda", "one_plus_lambda_lambda")


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


def _validate_algorithm(algorithm):
    """Return ``algorithm`` if it is one of :data:`ALGORITHMS`."""
    if algorithm not in ALGORITHMS:
        raise ValueError(
            f"Error: unknown algorithm {algorithm!r}; expected one of {ALGORITHMS}"
        )
    return algorithm


def _map_choice_to_index(env, choice, choices=None):
    """Map a portfolio value to the matching action index of ``env``.

    An exact hit returns its index; otherwise the nearest value in log space is
    used (the default portfolio is made of powers of two, so log space is the
    natural metric). ``choices`` defaults to ``env.action_choices``; without that
    list (continuous action space) the value is rounded to an integer.
    """
    if choices is None:
        choices = getattr(env, "action_choices", None)
    if choices is None:
        # continuous action space (only ``one_plus_one``): the policy values are
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


def theory_policy(problem, algorithm):
    """Return a theory-derived baseline policy for one combination.

    Stateless ``policy(env, state) -> int`` returning an action index, or the raw
    integer ``r`` on a continuous action space; it reads ``env.x.fitness``,
    ``env.n`` and ``env.action_choices``. This is a **baseline, not an optimum**:
    the published analyses bound the expected runtime of the parameter choice, they
    do not give the optimal finite-horizon policy.

    - ``one_plus_one`` / OneMax: ``r = 1`` (``p = 1/n``), the optimal rate.
    - ``one_plus_lambda`` / OneMax: ``lambda = 1``. ``lambda`` children cost
      ``lambda`` times as much and improve ``lambda`` times as often, so
      parallelism does not buy evaluations here; a heuristic, not a theorem.
    - ``one_plus_lambda_lambda`` / OneMax: ``round(sqrt(n / max(1, n - f(x))))``,
      mapped to the nearest portfolio value in log space and clipped at the top
      (the formula grows like ``sqrt(n)`` near the optimum).
    - ``one_plus_one`` / LeadingOne: ``r = 1``. The best published *fixed* rate is
      ``p ~ 1.59/n``, a nominal ``r = 1.59``, but this environment conditions on
      flipping at least one bit, which raises the expected number of flipped bits
      to ``r / (1 - exp(-r))``: 1.58 at ``r = 1`` and 2.31 at ``r = 2``. ``r = 1``
      therefore matches the published rate; paired measurements on the shipped
      ladders agree (730 instead of 1116 evaluations at n = 50, 3026 instead of
      4092 at n = 100).
    - ``one_plus_lambda`` / LeadingOne: ``lambda = 1``, the smallest admissible
      population (``lambda > 1`` multiplies the per-generation cost); no analysis
      covers this combination. It is a baseline, not an optimum: measured, a
      constant ``lambda = 2`` is cheaper on the LeadingOne ladders.
    - ``one_plus_lambda_lambda`` / LeadingOne: raises ``ValueError``, because there
      is no published analysis of that operator on LeadingOnes.

    Parameters
    ----------
    problem : str
        ``OneMax`` / ``LeadingOne`` or their file-name stems.
    algorithm : str
        One of :data:`ALGORITHMS`.

    Returns:
    --------
    callable
        ``policy(env, state) -> int`` with a no-op ``policy.reset()``.

    Raises:
    -------
    ValueError
        If the problem or the algorithm is unknown, or if no theory policy is
        available for the combination.

    """
    problem_stem = _problem_stem(problem)
    _validate_algorithm(algorithm)
    if problem_stem == "leadingones" and algorithm == "one_plus_lambda_lambda":
        raise ValueError(
            "Error: no theory policy for the (1+(lambda,lambda)) GA on "
            "LeadingOne: there is no published runtime analysis of that "
            "operator on LeadingOnes. Use success_rule, or measure the best "
            "fixed action in your run."
        )

    # constant action value, or ``None`` for the fitness-dependent GA rule
    constant = {
        ("onemax", "one_plus_one"): 1.0,
        ("onemax", "one_plus_lambda"): 1.0,
        ("onemax", "one_plus_lambda_lambda"): None,
        ("leadingones", "one_plus_one"): 1.0,
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


def success_rule(algorithm, action_choices=None, initial_value=1, factor=1.5):
    """Return a stateful one-fifth success rule policy.

    ``one_plus_one`` adapts the mutation strength ``r``, the population-based
    algorithms adapt ``lambda``:

    - ``one_plus_one``: ``r <- r * factor`` after an improvement, otherwise
      ``r <- r * factor**(-1/4)`` -- the multiplicative form of the one-fifth
      success rule (Kern et al. 2004, Eq. 3; Rechenberg 1973), the standard
      discrete analogue for an integer ``r``.
    - ``one_plus_lambda_lambda``: ``lambda <- lambda / factor`` after an
      improvement, otherwise ``lambda <- lambda * factor**(1/4)`` (Doerr & Doerr,
      GECCO 2015). The direction is reversed because a larger population has a
      higher success probability.
    - ``one_plus_lambda``: the same lambda-update, labelled an *adaptation*: the
      published analysis covers the (1+(lambda,lambda)) GA, not the (1+lambda) EA.

    The tracked value is mapped to the nearest action index in log space and
    clipped to the portfolio. With ``factor = 1.5`` the mean update is neutral at a
    success rate of 1/5, hence the name. The rule is stateful: it compares the
    fitness of consecutive calls (the environment accepts ties, so fitness does
    not decrease within an episode) and resets to ``initial_value`` on a new
    episode, detected by ``env.c_step == 0`` or a lower fitness.

    Parameters
    ----------
    algorithm : str
        One of :data:`ALGORITHMS`; selects what is adapted and in which direction.
    action_choices : sequence | None
        Portfolio of admissible values; defaults to ``env.action_choices`` at call
        time. Without a portfolio (continuous action space) the adapted ``r`` is
        returned as an integer clipped to ``[1, n]``.
    initial_value : int | float
        Value the rule starts each episode with (default 1).
    factor : float
        Update strength ``F > 1`` (default 1.5).

    Returns:
    --------
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
        Seed of the policy's own generator, independent of the environment.

    Returns:
    --------
    callable
        ``policy(env, state) -> int``, uniform over the action indices, or over
        ``[1, n]`` on a continuous action space. Call ``policy.reset()`` at the
        start of every episode (as ``examples/run_ga_baselines.py`` does), which
        re-seeds to ``seed``; otherwise the action stream continues across
        episodes and two runs are only comparable if they reset identically.

    """
    rng = np.random.default_rng(seed)

    def reset():
        nonlocal rng
        rng = np.random.default_rng(seed)

    def policy(env, state):
        # API: env.action_choices, env.n
        choices = getattr(env, "action_choices", None)
        if choices is None:
            # continuous action space (only ``one_plus_one``): sample r in [1, n]
            n = max(1, int(getattr(env, "n", 1) or 1))
            return int(rng.integers(1, n + 1))
        return int(rng.integers(len(choices)))

    policy.reset = reset
    return policy
