import logging
import uuid
from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np

from dacbench import AbstractEnv


class BinaryProblem:
    """An abstract class for an individual in binary representation."""

    def __init__(self, n, rng=np.random.default_rng()):
        """Init problem."""
        self.data = rng.choice([True, False], size=n)
        self.n = n
        self.fitness = self.eval()

    def initialise_with_fixed_number_of_bits(self, k, rng=np.random.default_rng()):
        """Init with given number of bits."""
        nbits = self.data.sum()
        if nbits < k:
            ids = rng.choice(
                np.where(self.data is False)[0], size=k - nbits, replace=False
            )
            self.data[ids] = True
            self.eval()

    def is_optimal(self):
        """Get is_optimal flag."""
        pass

    def get_optimal(self):
        """Get optimum."""
        pass

    def eval(self):
        """Evaluate fitness."""
        pass

    def get_fitness_after_flipping(self, locs):
        """
        Calculate the change in fitness after flipping the bits at positions locs

        Parameters
        ----------
        locs: 1d-array
            positions where bits are flipped

        Returns
        -------
            objective after flipping

        """
        raise NotImplementedError

    def get_fitness_after_crossover(self, xprime, locs_x, locs_xprime):
        """
        Calculate fitness of the child aftering being crossovered with xprime.

        Parameters
        ----------
        xprime: 1d boolean array
            the individual to crossover with
        locs_x: 1d boolean/integer array
            positions where we keep current bits of self
        locs_xprime: : 1d boolean/integer array
            positions where we change to xprime's bits

        Returns
        -------
            fitness of the new individual after crossover

        """
        raise NotImplementedError

    def flip(self, locs):
        """
        Flip the bits at position indicated by locs.

        Parameters
        ----------
        locs: 1d-array
            positions where bits are flipped

        Returns
        -------
            the new individual after the flip

        """
        child = deepcopy(self)
        child.data[locs] = ~child.data[locs]
        child.eval()
        return child

    def combine(self, xprime, locs_xprime):
        """
        Combine (crossover) self and xprime by taking xprime's bits at locs_xprime and self's bits at other positions.

        Parameters
        ----------
        xprime: 1d boolean array
            the individual to crossover with
        locs_x: 1d boolean/integer array
            positions where we keep current bits of self
        locs_xprime: : 1d boolean/integer array
            positions where we change to xprime's bits

        Returns
        -------
            the new individual after the crossover

        """
        child = deepcopy(self)
        child.data[locs_xprime] = xprime.data[locs_xprime]
        child.eval()
        return child

    def mutate(self, p, n_childs, rng=np.random.default_rng()):
        """
        Draw l ~ binomial(n, p), l>0.

        Generate n_childs children by flipping exactly l bits

        Returns
        -------
            the best child (maximum fitness), its fitness and number of evaluations used

        """
        assert p >= 0

        if p == 0:
            return self, self.fitness, 0

        l = 0
        while l == 0:
            l = rng.binomial(self.n, p)

        best_obj = -1
        best_locs = None
        for i in range(n_childs):
            locs = rng.choice(self.n, size=l, replace=False)
            obj = self.get_fitness_after_flipping(locs)
            if obj > best_obj:
                best_locs = locs
                best_obj = obj

        best_child = self.flip(best_locs)

        return best_child, best_child.fitness, n_childs

    def mutate_rls(self, l, rng=np.random.default_rng()):
        """
        Generate a child by flipping exactly l bits.

        Returns
        -------
            child, its fitness

        """
        assert l >= 0

        if l == 0:
            return self, self.fitness, 0

        locs = rng.choice(self.n, size=l, replace=False)
        child = self.flip(locs)

        return child, child.fitness, 1

    def crossover(
        self,
        xprime,
        p,
        n_childs,
        include_xprime=True,
        count_different_inds_only=True,
        rng=np.random.default_rng(),
    ):
        """
        Crossover operation in population.

        Crossover operator: for each bit, taking value from x with probability p and from self with probability 1-p

        Parameters
        ----------
        xprime
            the individual to crossover with
        p : float
            probability in [0,1]
        n_childs : int
            number of child individuals
        include_xprime : bool
            whether to inculde x
        count_different_inds_only : bool
            whether to only count different individuals
        rng:
            random number generator

        """
        assert p <= 1

        if p == 0:
            if include_xprime:
                return xprime, xprime.fitness, 0
            else:
                return self, self.fitness, 0

        if include_xprime:
            best_obj = xprime.fitness
        else:
            best_obj = -1
        best_locs = None

        n_evals = 0
        ls = rng.binomial(self.n, p, size=n_childs)
        for l in ls:
            locs_xprime = rng.choice(self.n, l, replace=False)
            locs_x = np.full(self.n, True)
            locs_x[locs_xprime] = False
            obj = self.get_fitness_after_crossover(xprime, locs_x, locs_xprime)

            if (obj != self.fitness) and (obj != xprime.fitness):
                n_evals += 1
            elif (
                not np.array_equal(xprime.data[locs_xprime], self.data[locs_xprime])
            ) and (not np.array_equal(self.data[locs_x], xprime.data[locs_x])):
                n_evals += 1

            if obj > best_obj:
                best_obj = obj
                best_locs = locs_xprime

        if best_locs is not None:
            child = self.combine(xprime, best_locs)
        else:
            child = xprime

        if not count_different_inds_only:
            n_evals = n_childs

        return child, child.fitness, n_evals


class LeadingOne(BinaryProblem):
    """
    An individual for LeadingOne problem.

    The aim is to maximise the number of leading (and consecutive) 1 bits in the string
    """

    def __init__(self, n, rng=np.random.default_rng(), initObj=None):
        """Make individual"""
        if initObj is None:
            super(LeadingOne, self).__init__(n=n, rng=rng)
        else:
            self.data = rng.choice([True, False], size=n)
            self.data[: int(initObj)] = True
            self.data[int(initObj)] = False
            self.n = n
            self.fitness = self.eval()

    def eval(self):
        """Evaluate fitness."""
        k = self.data.argmin()
        if self.data[k]:
            self.fitness = self.n
        else:
            self.fitness = k
        return self.fitness

    def is_optimal(self):
        """Return is_optimal flag."""
        return self.data.all()

    def get_optimal(self):
        """Return optimum."""
        return self.n

    def get_fitness_after_flipping(self, locs):
        """Return fitness after flipping."""
        min_loc = locs.min()
        if min_loc < self.fitness:
            return min_loc
        elif min_loc > self.fitness:
            return self.fitness
        else:
            old_fitness = self.fitness
            self.data[locs] = ~self.data[locs]
            new_fitness = self.eval()
            self.data[locs] = ~self.data[locs]
            self.fitness = old_fitness
            return new_fitness

    def get_fitness_after_crossover(self, xprime, locs_x, locs_xprime):
        """Return fitness after crossover."""
        child = self.combine(xprime, locs_xprime)
        child.eval()
        return child.fitness


MAX_INT = 1e8
HISTORY_LENGTH = 5


class TheoryEnv(AbstractEnv):
    """
    Environment for RLS with step size.

    Current assumption: we only consider (1+1)-RLS, so there's only one parameter to tune (r)
    """

    def __init__(self, config, test_env=False) -> None:
        """
        Initialize TheoryEnv.

        Parameters
        ----------
        config : objdict
            Environment configuration
        test_env : bool
            whether to use test mode

        """
        super(TheoryEnv, self).__init__(config)
        self.logger = logging.getLogger(self.__str__())

        self.test_env = test_env

        self.name = config.name

        # name of reward function
        assert config.reward_choice in [
            "imp_div_evals",
            "imp_div_evals_new",
            "imp_minus_evals",
            "minus_evals",
            "imp",
            "minus_evals_normalised",
            "imp_minus_evals_normalised",
        ]
        self.reward_choice = config.reward_choice
        # print("Reward choice: " + self.reward_choice)

        # get problem
        self.problem = globals()[config.problem]

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
            elif var_name in ["r"]:
                self.state_functions.append(
                    lambda his="history_" + var_name: vars(self)[his][-1]
                )
            elif (
                "_{t-" in var_name
            ):  # TODO: this implementation only allow accessing history of r, but not delta_f(x), optimal_k, etc
                k = int(
                    var_name.split("_{t-")[1][:-1]
                )  # get the number in _{t-<number>}
                name = var_name.split("_{t-")[0]  # get the variable name (r, f(x), etc)
                self.state_functions.append(
                    lambda his="history_" + name: vars(self)[his][-(k + 1)]
                )  # the last element is the value at the current time step, so we have to go one step back to access the history
            elif var_name == "f(x)":
                self.state_functions.append(lambda: self.history_fx[-1])
            elif var_name == "delta_f(x)":
                self.state_functions.append(
                    lambda: self.history_fx[-1] - self.history_fx[-2]
                )
            elif var_name == "optimal_r":
                self.state_functions.append(
                    lambda: int(self.n / (self.history_fx[-1] + 1))
                )
            else:
                raise Exception("Error: invalid state variable name: " + var_name)

        # the random generator used by RLS
        if "seed" in config:
            seed = config.seed
        else:
            seed = None
        if "seed" in self.instance:
            seed = self.instance.seed
        self.seed(seed)

        # for logging
        self.outdir = None
        if "outdir" in config:
            self.outdir = config.outdir + "/" + str(uuid.uuid4())

    def get_obs_domain_from_name(var_name):
        """
        Get default lower and upperbound of a observation variable based on its name.

        The observation space will then be created

        Returns
        -------
            Two int values, e.g., 1, np.inf

        """
        return 0, np.inf

    def reset(self, seed=None, options={}):
        """
        Resets env.

        Returns
        -------
        numpy.array
            Environment state

        """
        super(TheoryEnv, self).reset_(seed)

        # current problem size (n) & evaluation limit (max_evals)
        self.n = self.instance.size
        if self.test_env:
            self.max_evals = self.n_steps
        else:
            self.max_evals = int(0.8 * self.n * self.n)
        self.logger.info("n:%d, max_evals:%d" % (self.n, self.max_evals))

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

        # reset histories
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

    def step(self, action):
        """
        Execute environment step.

        Parameters
        ----------
        action : Box
            action to execute

        Returns
        -------
            state, reward, terminated, truncated, info
            np.array, float, bool, bool, dict

        """
        truncated = super(TheoryEnv, self).step_()

        fitness_before_update = self.x.fitness

        # get r
        if isinstance(action, np.ndarray) or isinstance(action, list):
            assert len(action) == 1
            r = action[0]
        else:
            r = action

        # if r is out of range
        stop = False
        if r < 1 or r > self.n:
            self.logger.info(f"WARNING: r={r} is out of bound")

            # if we're in the training phase, we return a large negative reward and stop the episode
            if self.test_env is False:
                terminated = True
                n_evals = 0
                reward = -MAX_INT
                stop = True
            # if we're in the test phase, just clip r back to the range and continue
            else:
                r = np.clip(r, 1, self.n)

        if stop is False:
            # flip r bits
            r = int(r)
            y, f_y, n_evals = self.x.mutate_rls(r, self.np_random)

            # update x
            if self.x.fitness <= y.fitness:
                self.x = y

            # update total number of evaluations
            self.total_evals += n_evals

            # check stopping criteria
            terminated = (self.total_evals >= self.max_evals) or (self.x.is_optimal())

            # calculate reward
            if self.reward_choice == "imp_div_evals":
                reward = (self.x.fitness - fitness_before_update - 0.5) / n_evals
            elif self.reward_choice == "imp_minus_evals":
                reward = self.x.fitness - fitness_before_update - n_evals
            elif self.reward_choice == "minus_evals":
                reward = -n_evals
            elif self.reward_choice == "minus_evals_normalised":
                reward = -n_evals / self.max_evals
            elif self.reward_choice == "imp_minus_evals_normalised":
                reward = (
                    self.x.fitness - fitness_before_update - n_evals
                ) / self.max_evals
            elif self.reward_choice == "imp":
                reward = self.x.fitness - fitness_before_update - 0.5
            self.log_reward.append(reward)

        # update histories
        self.history_fx.append(self.x.fitness)
        self.history_r.append(r)

        # update logs
        self.log_r.append(r)
        self.log_fx.append(self.x.fitness)
        self.log_reward.append(reward)

        returned_info = {"msg": "", "values": {}}
        if terminated or truncated:
            if hasattr(self, "env_type"):
                msg = "Env " + self.env_type + ". "
            else:
                msg = ""
            msg += (
                "Episode done: n=%d; obj=%d; init_obj=%d; evals=%d; max_evals=%d; steps=%d; r_min=%.1f; r_max=%.1f; r_mean=%.1f; R=%.4f"
                % (
                    self.n,
                    self.x.fitness,
                    self.init_obj,
                    self.total_evals,
                    self.max_evals,
                    self.c_step,
                    min(self.log_r),
                    max(self.log_r),
                    sum(self.log_r) / len(self.log_r),
                    sum(self.log_reward),
                )
            )
            # self.logger.info(msg)
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

        return self.get_state(), reward, truncated, terminated, returned_info

    def close(self) -> bool:
        """
        Close Env.

        No additional cleanup necessary

        Returns
        -------
        bool
            Closing confirmation

        """
        return True


class TheoryEnvDiscrete(TheoryEnv):
    """RLS environment where the choices of r is discretised."""

    def __init__(self, config, test_env=False):
        """Init env."""
        super(TheoryEnvDiscrete, self).__init__(config, test_env)
        assert (
            "action_choices" in config
        ), "Error: action_choices must be specified in benchmark's config"
        assert isinstance(
            self.action_space, gym.spaces.Discrete
        ), "Error: action space must be discrete"
        assert self.action_space.n == len(config["action_choices"]), (
            "Error: action space's size (%d) must be equal to the len(action_choices) (%d)"
            % (self.action_space.n, len(config["action_choices"]))
        )
        self.action_choices = config["action_choices"]

    def step(self, action):
        """Take step."""
        if isinstance(action, np.ndarray) or isinstance(action, list):
            assert len(action) == 1
            action = action[0]
        return super(TheoryEnvDiscrete, self).step(self.action_choices[action])
