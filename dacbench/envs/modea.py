# Most of this code is taken from the paper "Online CMA-ES Selection" by Vermetten et al.
# Github: https://github.com/Dvermetten/Online_CMA-ES_Selection

from modea.Algorithms import CustomizedES
from modea.Parameters import Parameters
import modea.Sampling as Sam
import modea.Mutation as Mut
import modea.Selection as Sel
import modea.Recombination as Rec
from dacbench import AbstractEnv
from modea.Utils import getOpts, getVals, options, initializable_parameters
from cma import bbobbenchmarks as bn
from functools import partial
import numpy as np


class ModeaEnv(AbstractEnv):
    def __init__(self, config):
        super(ModeaEnv, self).__init__(config)
        self.es = None
        self.budget = config.budget
        self.total_budget = self.budget

        if "reward_function" in config.keys():
            self.get_reward = config["reward_function"]
        else:
            self.get_reward = self.get_default_reward

        if "state_method" in config.keys():
            self.get_state = config["state_method"]
        else:
            self.get_state = self.get_default_state

    def reset(self):
        super(ModeaEnv, self).reset_()
        self.dim = self.instance[0]
        self.function_id = self.instance[1]
        self.instance_id = self.instance[2]
        self.representation = self.ensureFullLengthRepresentation(self.instance[3])

        opts = getOpts(self.representation[: len(options)])
        self.lambda_ = self.representation[len(options)]
        self.mu = self.representation[len(options) + 1]
        values = getVals(self.representation[len(options) + 2 :])

        self.function, self.target = bn.instantiate(int(self.function_id))
        self.es = CustomizedES(
            self.dim, self.function, self.budget, self.mu, self.lambda_, opts, values
        )
        parameter_opts = self.es.parameters.getParameterOpts()
        # print("Local restart on")
        if parameter_opts["lambda_"]:
            self.lambda_init = parameter_opts["lambda_"]
        elif parameter_opts["local_restart"] in ["IPOP", "BIPOP"]:
            self.lambda_init = int(4 + np.floor(3 * np.log(parameter_opts["n"])))
        else:
            self.lambda_init = None
        parameter_opts["lambda_"] = self.lambda_init

        # BIPOP Specific parameters
        self.lambda_ = {"small": None, "large": self.lambda_init}
        self.budgets = {"small": None, "large": None}
        self.regime = "first"
        self.update_parameters()
        return self.get_state(self)

    def step(self, action):
        done = super(ModeaEnv, self).step_()
        self.representation = self.ensureFullLengthRepresentation(action)
        opts = getOpts(self.representation[: len(options)])
        self.switchConfiguration(opts)

        self.es.runOneGeneration()
        self.es.recordStatistics()

        if (
            self.es.budget <= self.es.used_budget
            or self.es.parameters.checkLocalRestartConditions(self.es.used_budget)
        ):
            done = done or self.restart()
            if self.es.total_used_budget < self.es.total_budget:
                self.update_parameters()
            else:
                done = True

        return self.get_state(self), self.get_reward(self), done, {}

    def update_parameters(self):
        # Every local restart needs its own parameters, so parameter update/mutation must also be linked every time
        parameter_opts = self.es.parameters.getParameterOpts()
        self.es.parameters = Parameters(**parameter_opts)
        self.es.seq_cutoff = self.es.parameters.mu_int * self.es.parameters.seq_cutoff
        self.es.mutateParameters = self.es.parameters.adaptCovarianceMatrix

        self.es.initializePopulation()
        parameter_opts["wcm"] = self.es.population[0].genotype
        self.es.new_population = self.es.recombine(
            self.es.population, self.es.parameters
        )

    def restart(self):
        done = False
        parameter_opts = self.es.parameters.getParameterOpts()
        self.es.total_used_budget += self.es.used_budget
        if self.target is not None:
            # TODO: make threshold an env parameter
            if self.es.best_individual.fitness - self.target <= 1e-8:
                done = True
        # Increasing Population Strategies
        if parameter_opts["local_restart"] == "IPOP":
            parameter_opts["lambda_"] *= 2

        elif parameter_opts["local_restart"] == "BIPOP":
            try:
                self.budgets[self.regime] -= self.es.used_budget
                self.determineRegime()
            except KeyError:  # Setup of the two regimes after running regularily for the first time
                remaining_budget = self.total_budget - self.es.used_budget
                self.budgets["small"] = remaining_budget // 2
                self.budgets["large"] = remaining_budget - self.budgets["small"]
                self.regime = "large"

            if self.regime == "large":
                self.lambda_["large"] *= 2
                parameter_opts["sigma"] = 2
            elif self.regime == "small":
                rand_val = self.np_random.random() ** 2
                self.lambda_["small"] = int(
                    np.floor(
                        self.lambda_init
                        * (0.5 * self.es.lambda_["large"] / self.lambda_init)
                        ** rand_val
                    )
                )
                parameter_opts["sigma"] = 2e-2 * self.np_random.random()

            self.es.budget = self.budgets[self.regime]
            self.es.used_budget = 0
            parameter_opts["budget"] = self.budget
            parameter_opts["lambda_"] = self.lambda_[self.regime]
        return done

    def determineRegime(self):
        large = self.budgets["large"]
        small = self.budgets["small"]
        if large <= 0:
            self.regime = "small"
        elif small <= 0:
            self.regime = "large"
        elif large > small:
            self.regime = "large"
        else:
            self.regime = "small"

    def get_default_state(self, _):
        return np.array(
            [
                self.es.gen_size,
                self.es.parameters.sigma,
                self.budget - self.es.used_budget,
                self.function_id,
                self.instance_id,
            ]
        )

    def get_default_reward(self, _):
        return max(
            self.reward_range[0],
            min(self.reward_range[1], -self.es.best_individual.fitness),
        )

    def close(self):
        return True

    def switchConfiguration(self, opts):
        selector = Sel.pairwise if opts["selection"] == "pairwise" else Sel.best

        def select(pop, new_pop, _, param):
            return selector(pop, new_pop, param)

        # Pick the lowest-level sampler
        if opts["base-sampler"] == "quasi-sobol":
            sampler = Sam.QuasiGaussianSobolSampling(self.es.n)
        elif opts["base-sampler"] == "quasi-halton" and Sam.halton_available:
            sampler = Sam.QuasiGaussianHaltonSampling(self.es.n)
        else:
            sampler = Sam.GaussianSampling(self.es.n)

        # Create an orthogonal sampler using the determined base_sampler
        if opts["orthogonal"]:
            orth_lambda = self.es.parameters.eff_lambda
            if opts["mirrored"]:
                orth_lambda = max(orth_lambda // 2, 1)
            sampler = Sam.OrthogonalSampling(
                self.es.n, lambda_=orth_lambda, base_sampler=sampler
            )

        # Create a mirrored sampler using the sampler (structure) chosen so far
        if opts["mirrored"]:
            sampler = Sam.MirroredSampling(self.es.n, base_sampler=sampler)

        parameter_opts = {
            "weights_option": opts["weights_option"],
            "active": opts["active"],
            "elitist": opts["elitist"],
            "sequential": opts["sequential"],
            "tpa": opts["tpa"],
            "local_restart": opts["ipop"],
        }

        # In case of pairwise selection, sequential evaluation may only stop after 2mu instead of mu individuals

        if opts["sequential"] and opts["selection"] == "pairwise":
            parameter_opts["seq_cutoff"] = 2
            self.es.parameters.seq_cutoff = 2

        # Init all individuals of the first population at the same random point in the search space

        # We use functions/partials here to 'hide' the additional passing of parameters that are algorithm specific
        recombine = Rec.weighted
        mutate = partial(
            Mut.CMAMutation, sampler=sampler, threshold_convergence=opts["threshold"]
        )

        functions = {
            "recombine": recombine,
            "mutate": mutate,
            "select": select,
            # 'mutateParameters': None
        }
        self.setConfigurationParameters(functions, parameter_opts)
        lambda_, eff_lambda, mu = self.es.calculateDependencies(opts, None, None)
        self.es.parameters.lambda_ = lambda_
        self.es.parameters.eff_lambda = eff_lambda
        self.es.parameters.mu = mu
        self.es.parameters.weights = self.es.parameters.getWeights(
            self.es.parameters.weights_option
        )
        self.es.parameters.mu_eff = 1 / np.sum(np.square(self.es.parameters.weights))
        mu_eff = self.es.parameters.mu_eff  # Local copy
        n = self.es.parameters.n
        self.es.parameters.c_sigma = (mu_eff + 2) / (mu_eff + n + 5)
        self.es.parameters.c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        self.es.parameters.c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        self.es.parameters.c_mu = min(
            1 - self.es.parameters.c_1,
            self.es.parameters.alpha_mu
            * (
                (mu_eff - 2 + 1 / mu_eff)
                / ((n + 2) ** 2 + float(self.es.parameters.alpha_mu) * mu_eff / 2)
            ),
        )
        self.es.parameters.damps = (
            1
            + 2 * np.max([0, np.sqrt((mu_eff - 1) / (n + 1)) - 1])
            + self.es.parameters.c_sigma
        )
        self.es.seq_cutoff = self.es.parameters.mu_int * self.es.parameters.seq_cutoff

    def setConfigurationParameters(self, functions, parameters):
        self.es.recombine = functions["recombine"]
        self.es.mutate = functions["mutate"]
        self.es.select = functions["select"]
        # self.mutateParameters = functions['mutateParameters']
        self.es.parameters.weights_option = parameters["weights_option"]
        self.es.parameters.active = parameters["active"]
        self.es.parameters.elitist = parameters["elitist"]
        self.es.parameters.sequential = parameters["sequential"]
        self.es.parameters.tpa = parameters["tpa"]
        self.es.parameters.local_restart = parameters["local_restart"]

    def ensureFullLengthRepresentation(self, representation):
        """
        Given a (partial) representation, ensure that it is padded to become a full length customizedES representation,
        consisting of the required number of structure, population and parameter values.
        >>> ensureFullLengthRepresentation([])
        [0,0,0,0,0,0,0,0,0,0,0, None,None, None,None,None,None,None,None,None,None,None,None,None,None,None]
        :param representation:  List representation of a customizedES instance to check and pad if needed
        :return:                Guaranteed full-length version of the representation
        """
        default_rep = (
            [0] * len(options) + [None, None] + [None] * len(initializable_parameters)
        )
        if len(representation) < len(default_rep):
            representation = np.append(
                representation, default_rep[len(representation) :]
            ).flatten()
        return representation
