# Portions of this file are adapted from
# https://github.com/taindp98/OneMax-DAC-Extension (dacbench/envs/theory.py).
#
# The MIT License (MIT)
# Copyright (c) 2025, xxx
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Modified 2026 for automl/DACBench (Apache-2.0).
"""GA Environment: (1+1) EA, (1+lambda) EA and (1+(lambda,lambda)) GA.

Dynamic algorithm configuration of simple evolutionary algorithms on the bitstring
problems OneMax and LeadingOnes. One environment step corresponds to one generation
of the selected algorithm; the action controls the mutation strength r of the
(1+1) EA or the population size lambda of the (1+lambda) EA / (1+(lambda,lambda)) GA.

The (1+(lambda,lambda)) GA environment is adapted from the OneMax DAC extension of
DACBench by Tai Nguyen, Phong Le, André Biedenkapp, Carola Doerr and Nguyen Dang
(https://github.com/taindp98/OneMax-DAC-Extension), see arXiv:2502.20265,
arXiv:2505.12982 and arXiv:2606.10129.

The (1+(lambda,lambda)) GA operator follows B. Doerr, C. Doerr, F. Ebel. "From
black-box complexity to designing new genetic algorithms". Theoretical Computer
Science, 2015. Evaluations are counted in the practice-aware way of E. Carvalho
Pinto, C. Doerr. "Towards a more practice-aware runtime analysis of evolutionary
algorithms", 2018 (mutation strength resampled until > 0, crossover offspring
identical to one of their parents are not evaluated).
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

from dacbench import AbstractEnv
from dacbench.envs.theory import LeadingOne, OneMax

__copyright__ = (
    "Copyright (c) 2025, xxx (adapted portions); modifications Copyright 2026, "
    "automl.org"
)
__license__ = "MIT (adapted portions); Apache-2.0 (DACBench)"

MAX_INT = 1e8
HISTORY_LENGTH = 5

PROBLEMS = {"OneMax": OneMax, "LeadingOne": LeadingOne}

ALGORITHMS = ("one_plus_one", "one_plus_lambda", "one_plus_lambda_lambda")

REWARD_CHOICES = (
    "imp_div_evals",
    "imp_div_evals_new",
    "imp_minus_evals",
    "minus_evals",
    "imp",
    "minus_evals_normalised",
    "imp_minus_evals_normalised",
    "imp_minus_evals_scaling",
)

# observation variables: name -> (lower bound, upper bound)
OBS_DOMAINS = {
    "n": (0, np.inf),
    "f(x)": (0, np.inf),
    "delta_f(x)": (0, np.inf),
    "f(x)/n": (0, 1),
    "r": (0, np.inf),
    "lambda": (0, np.inf),
}
# variables with a history that can be accessed via <name>_{t-k}, k < HISTORY_LENGTH
# ("r" and "lambda" are aliases of the same history: the last action value)
HISTORY_NAMES = {"r": "history_r", "lambda": "history_r", "f(x)": "history_fx"}


def _parse_history_name(var_name):
    """Split '<name>_{t-<k>}' into (history attribute name, k); raise if invalid."""
    name, _, rest = var_name.partition("_{t-")
    if name not in HISTORY_NAMES or not rest.endswith("}") or not rest[:-1].isdigit():
        raise ValueError("Error: invalid state variable name: " + var_name)
    k = int(rest[:-1])
    if k >= HISTORY_LENGTH:
        raise ValueError(
            f"Error: invalid state variable name: {var_name} "
            f"(history length is {HISTORY_LENGTH})"
        )
    return HISTORY_NAMES[name], k


def mutate_independent(x, p, n_childs, rng):
    """Create n_childs children, each with its own mutation strength.

    Each child independently draws l_i ~ Bin(n, p) conditioned on l_i > 0
    (resampled until > 0) and flips l_i uniformly chosen bits of x. Ties among
    the children with the best fitness are broken uniformly at random (reservoir
    sampling with rng). This differs from BinaryProblem.mutate, where all children
    share the same l.

    Returns:
    --------
        the best child, its fitness and the number of evaluations (n_childs)

    """
    best_obj = -1
    best_locs = None
    n_ties = 0
    for _ in range(n_childs):
        length = 0
        while length == 0:
            length = rng.binomial(x.n, p)
        locs = rng.choice(x.n, size=length, replace=False)
        obj = x.get_fitness_after_flipping(locs)
        if obj > best_obj:
            best_obj, best_locs, n_ties = obj, locs, 1
        elif obj == best_obj:
            n_ties += 1
            if rng.random() < 1.0 / n_ties:
                best_locs = locs
    child = x.flip(best_locs)
    return child, child.fitness, n_childs


class GAEnv(AbstractEnv):
    """Environment for the (1+1) EA, (1+lambda) EA and (1+(lambda,lambda)) GA.

    Config key ``algorithm`` selects the algorithm; one step is one generation:

    - ``one_plus_one``: action r (mutation strength). One child y is created with
      standard bit mutation at rate r/n, conditioned on flipping at least one bit
      ((1+1) EA_{>0}); y replaces x if f(y) >= f(x). Uses 1 evaluation.
    - ``one_plus_lambda``: action lambda. lambda children, each with its own
      l_i ~ Bin_{>0}(n, 1/n) flipped bits; the best child (uniform tie-breaking)
      replaces x if it is at least as good. Uses lambda evaluations.
    - ``one_plus_lambda_lambda``: action lambda. Mutation phase: l ~ Bin_{>0}(n, p)
      with p = lambda/n, lambda mutants flipping l bits each, x' = best mutant.
      Crossover phase: lambda offspring taking each bit from x' with probability
      c = 1/lambda (else from x); y = best offspring, where x' is the incumbent and
      only replaced by a strictly better offspring. y replaces x if
      f(y) >= f(x) (Doerr, Doerr & Ebel 2015). Evaluations are counted in the
      practice-aware way (Carvalho Pinto & Doerr 2018): crossover offspring
      identical to x or x' are not evaluated (``count_different_inds_only=True``,
      default), so one generation uses at most 2*lambda evaluations, and exactly
      2*lambda only with ``count_different_inds_only=False``.

    If ``discrete_action`` is True, the action is an index into
    ``action_choices``; otherwise (only for ``one_plus_one``) it is the raw r.
    Out-of-range values (< 1 or > n, including NaN and +-inf) and invalid indices
    end the episode with a reward of -MAX_INT in a training env; in a test env
    they are clipped (+-inf to [1, n]), except NaN, which raises a ValueError.
    """

    def __init__(self, config, test_env=False) -> None:
        """Initialize GAEnv.

        Parameters
        ----------
        config : objdict
            Environment configuration
        test_env : bool
            whether to use test mode

        """
        super().__init__(config)
        self.logger = logging.getLogger(self.__str__())

        self.test_env = test_env

        self.name = config.name

        # algorithm and action handling
        self.algorithm = config.algorithm
        assert self.algorithm in ALGORITHMS, (
            f"Error: algorithm must be one of {ALGORITHMS}, got {self.algorithm}"
        )
        self.discrete_action = config.discrete_action
        if not self.discrete_action and self.algorithm != "one_plus_one":
            raise ValueError(
                f"Error: algorithm {self.algorithm} controls the population size "
                "lambda and requires a discrete action space "
                "(discrete_action=True with action_choices)."
            )
        if self.discrete_action:
            assert "action_choices" in config, (
                "Error: action_choices must be specified in benchmark's config"
            )
            self.action_choices = list(config.action_choices)
        self.count_different_inds_only = config.get("count_different_inds_only", True)

        # name of reward function
        if config.reward_choice not in REWARD_CHOICES:
            raise ValueError(
                f"Error: reward_choice must be one of {REWARD_CHOICES}, "
                f"got {config.reward_choice}"
            )
        self.reward_choice = config.reward_choice

        # get problem
        assert config.problem in PROBLEMS, (
            f"Error: problem must be one of {list(PROBLEMS)}, got {config.problem}"
        )
        self.problem = PROBLEMS[config.problem]

        # read names of all observation variables
        self.obs_description = config.observation_description
        self.obs_var_names = [
            s.strip() for s in config.observation_description.split(",")
        ]

        # functions to get values of the current state from histories
        # (see reset() function for those history variables)
        self.state_functions = []
        for var_name in self.obs_var_names:
            if var_name == "n":
                self.state_functions.append(lambda: self.n)
            elif var_name in ["r", "lambda"]:
                self.state_functions.append(lambda: self.history_r[-1])
            elif "_{t-" in var_name:
                his, k = _parse_history_name(var_name)
                # the last element is the value at the current time step
                self.state_functions.append(
                    lambda k=k, his=his: vars(self)[his][-(k + 1)]
                )
            elif var_name == "f(x)":
                self.state_functions.append(lambda: self.history_fx[-1])
            elif var_name == "delta_f(x)":
                self.state_functions.append(
                    lambda: self.history_fx[-1] - self.history_fx[-2]
                )
            elif var_name == "f(x)/n":
                self.state_functions.append(lambda: self.history_fx[-1] / self.n)
            else:
                raise ValueError("Error: invalid state variable name: " + var_name)

        # the random generator used by the algorithms
        seed = config.seed if "seed" in config else None
        if "seed" in self.instance:
            seed = self.instance.seed
        self.seed(seed)

    @staticmethod
    def get_obs_domain_from_name(var_name):
        """Get default lower and upperbound of a observation variable based on its name.

        The observation space will then be created

        Returns:
        --------
        Two values, e.g., 0, np.inf
        """
        if var_name in OBS_DOMAINS:
            return OBS_DOMAINS[var_name]
        if "_{t-" in var_name:
            _parse_history_name(var_name)
            return 0, np.inf
        raise ValueError("Error: invalid state variable name: " + var_name)

    def get_ollga_params(self, lbd):
        """Return (mutation rate p, crossover bias c) = (lambda/n, 1/lambda)."""
        return lbd / self.n, 1.0 / lbd

    def reset(self, seed=None, options=None):
        """Resets env.

        Returns:
        --------
        numpy.array: Environment state
        """
        if options is None:
            options = {}
        super().reset_(seed, options)

        # current problem size (n) & evaluation limit (max_evals)
        self.n = self.instance.size
        if self.test_env:
            self.max_evals = self.n_steps
        else:
            self.max_evals = int(0.8 * self.n * self.n)
        self.logger.info("n:%d, max_evals:%d", self.n, self.max_evals)

        # set random seed
        if "seed" in self.instance:
            self.seed(self.instance.seed)

        # create an initial solution
        if self.instance.initObj == "random":
            self.x = self.problem(n=self.instance.size, rng=self.np_random)
        else:
            self.x = self.problem(
                n=self.instance.size, rng=self.np_random, initObj=self.instance.initObj
            )

        # total number of evaluations so far
        self.total_evals = 1

        # reset histories (history_r holds the last action values, r or lambda)
        self.history_r = deque([0] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
        self.history_fx = deque(
            [self.x.fitness] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH
        )

        # for debug only
        self.log_r = []
        self.log_reward = []
        self.log_fx = []
        self.init_obj = self.x.fitness

        return self.get_state(), {}

    def get_state(self):
        """Return state."""
        return np.asarray([f() for f in self.state_functions])

    def _get_action_value(self, action):
        """Map an action to its value (r or lambda).

        Returns:
        --------
        value, valid: the action value (None if the index is invalid in a training
        env) and whether the action was valid
        """
        if isinstance(action, list | tuple | np.ndarray):
            action = np.asarray(action).reshape(-1)
            assert len(action) == 1, "Error: action must be a single value"
            action = action[0]
        if not self.discrete_action:
            return float(action), True
        if float(action) != int(action):
            raise ValueError(f"Error: discrete action must be an integer: {action}")
        index = int(action)
        if 0 <= index < len(self.action_choices):
            return self.action_choices[index], True
        self.logger.info(f"WARNING: action index {index} is out of bound")
        if self.test_env is False:
            return None, False
        index = int(np.clip(index, 0, len(self.action_choices) - 1))
        return self.action_choices[index], True

    def _generation(self, value):
        """Run one generation with action value r or lambda; return n_evals."""
        if self.algorithm == "one_plus_one":
            y, _, n_evals = self.x.mutate(
                p=value / self.n, n_childs=1, rng=self.np_random
            )
        elif self.algorithm == "one_plus_lambda":
            y, _, n_evals = mutate_independent(
                self.x, p=1.0 / self.n, n_childs=value, rng=self.np_random
            )
        else:
            p, c = self.get_ollga_params(value)
            xprime, _, ne1 = self.x.mutate(p=p, n_childs=value, rng=self.np_random)
            y, _, ne2 = self.x.crossover(
                xprime,
                c,
                n_childs=value,
                include_xprime=True,
                count_different_inds_only=self.count_different_inds_only,
                rng=self.np_random,
            )
            n_evals = ne1 + ne2
        if self.x.fitness <= y.fitness:
            self.x = y
        return n_evals

    def _get_reward(self, fitness_before_update, n_evals):
        """Compute the reward of a step."""
        imp = self.x.fitness - fitness_before_update
        if self.reward_choice in ("imp_div_evals", "imp_div_evals_new"):
            # both upstream variants use the same formula; n_evals >= 1 for all
            # algorithms, max() only guards the division
            return (imp - 0.5) / max(n_evals, 1)
        if self.reward_choice == "imp_minus_evals":
            return imp - n_evals
        if self.reward_choice == "minus_evals":
            return -n_evals
        if self.reward_choice == "minus_evals_normalised":
            return -n_evals / self.max_evals
        if self.reward_choice == "imp_minus_evals_normalised":
            return (imp - n_evals) / self.max_evals
        if self.reward_choice == "imp":
            return imp - 0.5
        # imp_minus_evals_scaling
        return (imp - n_evals) / self.n

    def step(self, action):
        """Execute environment step.

        Parameters
        ----------
        action : int | float | list | np.ndarray
            index into action_choices (discrete) or raw r (continuous)

        Returns:
        --------
        np.array, float, bool, bool, dict: state, reward, terminated, truncated, info
        (correct Gymnasium order; note that TheoryEnv returns truncated and
        terminated swapped)
        """
        truncated = super().step_()

        fitness_before_update = self.x.fitness

        value, valid = self._get_action_value(action)

        # NaN cannot be clipped to a meaningful value in a test env
        if valid and self.test_env and np.isnan(value):
            raise ValueError(f"Error: action value must not be NaN: {action}")

        # if the action (index or value) is out of range (incl. non-finite values)
        stop = False
        if not valid or not np.isfinite(value) or value < 1 or value > self.n:
            self.logger.info(f"WARNING: action value {value} is out of bound")

            # if we're in the training phase, we return a large negative reward
            # and stop the episode
            if self.test_env is False:
                terminated = True
                reward = -MAX_INT
                stop = True
                if value is None or not np.isfinite(value):
                    value = 0  # invalid index / non-finite: no action value to log
            # if we're in the test phase, just clip back to the range and continue
            else:
                value = np.clip(value, 1, self.n)

        if stop is False:
            value = int(value)
            n_evals = self._generation(value)

            # update total number of evaluations
            self.total_evals += n_evals

            # check stopping criteria
            terminated = (self.total_evals >= self.max_evals) or (self.x.is_optimal())

            reward = self._get_reward(fitness_before_update, n_evals)

        # update histories
        self.history_fx.append(self.x.fitness)
        self.history_r.append(value)

        # update logs
        self.log_r.append(value)
        self.log_fx.append(self.x.fitness)
        self.log_reward.append(reward)

        returned_info = {"msg": "", "values": {}}
        if terminated or truncated:
            msg = "Env " + self.env_type + ". " if hasattr(self, "env_type") else ""
            msg += (
                f"Episode done: n={self.n}; obj={self.x.fitness}; "
                f"init_obj={self.init_obj}; evals={int(self.total_evals)}; "
                f"max_evals={int(self.max_evals)}; steps={self.c_step}; "
                f"r_min={min(self.log_r):.1f}; r_max={max(self.log_r):.1f}; "
                f"r_mean={sum(self.log_r) / len(self.log_r):.1f}; "
                f"R={sum(self.log_reward):.4f}"
            )
            returned_info["msg"] = msg
            returned_info["values"] = {
                "n": int(self.n),
                "obj": int(self.x.fitness),
                "init_obj": int(self.init_obj),
                "evals": int(self.total_evals),
                "max_evals": int(self.max_evals),
                "steps": int(self.c_step),
                "r_min": float(min(self.log_r)),
                "r_max": float(max(self.log_r)),
                "r_mean": float(sum(self.log_r) / len(self.log_r)),
                "R": float(sum(self.log_reward)),
                "log_r": [int(x) for x in self.log_r],
                "log_fx": [int(x) for x in self.log_fx],
                "log_reward": [float(x) for x in self.log_reward],
            }

        return self.get_state(), reward, terminated, truncated, returned_info

    def close(self) -> bool:
        """Close Env.

        No additional cleanup necessary

        Returns:
        --------
        bool: Closing confirmation
        """
        return True
