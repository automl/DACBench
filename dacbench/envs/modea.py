from modea.Algorithms import CustomizedES
import modea.Sampling as Sam
import modea.Mutation as Mut
import modea.Selection as Sel
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

        self.function = bn.instantiate(int(self.function_id))[0]
        self.es = CustomizedES(
            self.dim, self.function, self.budget, self.mu, self.lambda_, opts, values
        )
        self.es.mutateParameters = self.es.parameters.adaptCovarianceMatrix
        self.adapt_es_opts(opts)
        return self.get_state()

    def step(self, action):
        done = super(ModeaEnv, self).step_()
        # Todo: currently this doesn't support targets
        if (
            self.budget <= self.es.used_budget
            or self.es.parameters.checkLocalRestartConditions(self.es.used_budget)
        ):
            done = True

        self.representation = self.ensureFullLengthRepresentation(action)
        opts = getOpts(self.representation[: len(options)])
        self.adapt_es_opts(opts)

        # TODO: add ipop run (restarts)
        self.es.runOneGeneration()
        self.es.recordStatistics()

        return self.get_state(), self.get_reward(), done, {}

    def get_state(self):
        return [
            self.es.gen_size,
            self.es.parameters.sigma,
            self.budget - self.es.used_budget,
            self.function_id,
            self.instance_id,
        ]

    def get_reward(self):
        return max(
            self.reward_range[0],
            min(self.reward_range[1], -self.es.best_individual.fitness),
        )

    def close(self):
        return True

    def adapt_es_opts(self, opts):
        self.es.opts = opts
        parameter_opts = self.es.parameters.getParameterOpts()

        # __init__ of CustomizedES without new instance of ES
        # not a great solution, if package gets updates we should change this
        lambda_, eff_lambda, mu = self.es.calculateDependencies(
            opts, self.lambda_, self.mu
        )

        selector = Sel.pairwise if opts["selection"] == "pairwise" else Sel.best
        # This is done for safety reasons
        # Else offset and all_offspring may be None
        self.es.parameters.offset = np.column_stack(
            [ind.mutation_vector for ind in self.es.new_population]
        )
        self.es.parameters.all_offspring = np.column_stack(
            [ind.genotype for ind in self.es.new_population]
        )
        # Same here. Probably the representation space should be restricted
        self.es.parameters.tpa_result = -1

        def select(pop, new_pop, _, param):
            return selector(pop, new_pop, param)

        if opts["base-sampler"] == "quasi-sobol":
            sampler = Sam.QuasiGaussianSobolSampling(self.dim)
        elif opts["base-sampler"] == "quasi-halton" and Sam.halton_available:
            sampler = Sam.QuasiGaussianHaltonSampling(self.dim)
        else:
            sampler = Sam.GaussianSampling(self.dim)

        if opts["orthogonal"]:
            orth_lambda = eff_lambda
            if opts["mirrored"]:
                orth_lambda = max(orth_lambda // 2, 1)
            sampler = Sam.OrthogonalSampling(
                self.dim, lambda_=orth_lambda, base_sampler=sampler
            )

        if opts["mirrored"]:
            sampler = Sam.MirroredSampling(self.dim, base_sampler=sampler)

        if opts["sequential"] and opts["selection"] == "pairwise":
            parameter_opts["seq_cutoff"] = 2

        mutate = partial(
            Mut.CMAMutation, sampler=sampler, threshold_convergence=opts["threshold"]
        )

        self.es.mutate = mutate
        self.es.parameters = self.es.instantiateParameters(parameter_opts)
        self.es.seq_cutoff = self.es.parameters.mu_int * self.es.parameters.seq_cutoff

    # Source: https://github.com/sjvrijn/ConfiguringCMAES/blob/master/EvolvingES.py
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
