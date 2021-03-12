import numpy as np
import cocoex
import itertools
import pandas as pd
import random
from config import *
from copy import copy
from helpers import _init_cma_es, _update_cma_params, _extract_state_variables, _calculate_feature_set
from bbob import bbobbenchmarks
from pyDOE import lhs
from sobol_seq import i4_sobol
from pflacco.pflacco import create_feature_object
from dacbench import AbstractEnv
from dac_ela.helpers import _extract_state_variables

class CMAStepSizeEnv(AbstractEnv):
    def __init__(self, config):
        super().__init__(config)
        self.b = None
        self.bounds = [None, None]
        self.fbest = None
        self.history_len = config.hist_length
        self.history = deque(maxlen=self.history_len)
        self.past_obj_vals = deque(maxlen=self.history_len)
        self.past_sigma = deque(maxlen=self.history_len)
        self.solutions = None
        self.func_values = []
        self.cur_obj_val = -1
        self.popsize = config["popsize"]
        self.default_target = config["target"]
        self.cur_ps = self.popsize

        self.es = None
        self.budget = config.budget
        self.total_budget = self.budget
        self.sampling_strategy = config.sampling_strategy
        #TODO: how to record individuals?
        self.individuals = [[], []]
        self.num_restarts = None
        self.num_generations = 0
        self.sample_size = config.sample_size
        self.sobol_seed = config.sobol_seed

        self.get_reward = config.get("reward_function", self.get_default_reward)
        self.get_state = config.get("state_method", self.get_default_state)

    def reset(self):
        super().reset_()
        self.num_restarts = 0
        self.num_generations = 0
        self.dim, self.fid, self.iid, self.representation = self.instance
        self.objective = bbobbenchmarks.instantiate(self.fid, iinstance=self.iid)[1]
        self.mean = (np.random.rand(self.dim, 1) * 10) - 5
        self.representation = getOpts([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.es = _init_cma_es(ndim = self.dim, fitness_function = problem, budget = self.budget * self.dim - self.total_budget, opts = self.representation, mean = self.mean)
        parameter_opts = custom_es.parameters.getParameterOpts()
        return self.get_state(self)

    def step(self, action):
        done = super().step_()
        self.es.parameters.sigma = action
        # Run generation
        self.num_generations += 1
        self.es.runOneGeneration(self.individuals)
        self.es.recordStatistics()

        # This chunk of code is responsible for selecting the better individual found in the last generation in the case
        # where the population was only partially realized (due to budget restraints) and the evaluation had to be aborted
        if self.es.used_budget == self.es.budget:
            self.es.population = self.es.select(self.es.population, self.es.new_population,
                                                    self.es.used_budget, self.es.parameters)
        if self.es.population[0].fitness < self.es.best_individual.fitness:
            self.es.best_individual = self.es.population[0]

        # Update budget
        self.total_budget += self.es.used_budget
        # Check for restart
        if self.es.used_budget > self.es.budget \
                or self.es.parameters.checkLocalRestartConditions(self.es.used_budget) \
                or custom_es.best_individual.fitness - f_target <= self.default_target:
            self.es, self.representation = _update_cma_params(elf.es, self.representation, self.mean, self.num_restarts, (self.budget * self.dim) - self.total_budget)
            self.mean = (np.random.rand(self.dim, 1) * 10) - 5
            self.num_restarts += 1

        # Check if we move on
        if self.total_budget >= self.budget * self.dim or custom_es.best_individual.fitness - f_target <= self.default_target:
            done =  True

        return self.get_state(self), self.get_reward(self), done, {}

    def close(self):
        return True

    def get_default_reward(self, *_):
        return max(self.reward_range[0],
                   min(self.reward_range[1], -self.es.parameters.fopt)
                   )

    def get_default_state(self, _):
        """
        Gather state description

        Returns
        -------
        dict
            Environment state

        """
        past_obj_val_deltas = []
        for i in range(1, len(self.past_obj_vals)):
            past_obj_val_deltas.append(
                (self.past_obj_vals[i] - self.past_obj_vals[i - 1] + 1e-3)
                / float(self.past_obj_vals[i - 1])
            )
        if len(self.past_obj_vals) > 0:
            past_obj_val_deltas.append(
                (self.cur_obj_val - self.past_obj_vals[-1] + 1e-3)
                / float(self.past_obj_vals[-1])
            )
        past_obj_val_deltas = np.array(past_obj_val_deltas).reshape(-1)

        history_deltas = []
        for i in range(len(self.history)):
            history_deltas.append(self.history[i])
        history_deltas = np.array(history_deltas).reshape(-1)
        past_sigma_deltas = []
        for i in range(len(self.past_sigma)):
            past_sigma_deltas.append(self.past_sigma[i])
        past_sigma_deltas = np.array(past_sigma_deltas).reshape(-1)
        past_obj_val_deltas = np.hstack(
            (
                np.zeros((self.history_len - past_obj_val_deltas.shape[0],)),
                past_obj_val_deltas,
            )
        )
        history_deltas = np.hstack(
            (
                np.zeros((self.history_len * 2 - history_deltas.shape[0],)),
                history_deltas,
            )
        )
        past_sigma_deltas = np.hstack(
            (
                np.zeros((self.history_len - past_sigma_deltas.shape[0],)),
                past_sigma_deltas,
            )
        )

        cur_loc = np.array(self.cur_loc)
        cur_ps = np.array([self.cur_ps])
        cur_sigma = np.array(self.cur_sigma)

        state = {
            "current_loc": cur_loc,
            "past_deltas": past_obj_val_deltas,
            "current_ps": cur_ps,
            "current_sigma": cur_sigma,
            "history_deltas": history_deltas,
            "past_sigma_deltas": past_sigma_deltas,
        }
        return state

    def get_ela_state(self, *_):
        state_vars = _extract_state_variables(self.es, self.individuals, self.num_restarts, self.num_generations)
        # create sample based on CMA-ES distribution
        if self.sampling_strategy == 'cma-es':
            sample = np.random.multivariate_normal(self.es.parameters.wcm.flatten(),
                                                   self.es.parameters.sigma * self.es.parameters.C, self.sample_size)
        # create sample based on LHS
        elif self.sampling_strategy == 'lhs':
            sample = (lhs(self.dim,
                          samples=self.sample_size) - 0.5) * self.es.parameters.sigma + self.es.parameters.wcm.flatten()
        # create sample based on a Sobol sequence
        elif self.sampling_strategy == 'sobol':
            sample = []
            for x in range(self.sample_size):
                vec, seed = i4_sobol(self.dim, self.sobol_seed)
                sample.append(vec)
            sample = (np.array(sample) - 0.5) * self.es.parameters.sigma + self.es.parameters.wcm.flatten()
        # raise exception otherwise
        else:
            raise Exception(f'[{self.sampling_strategy}] is not a valid sampling strategy.')

        # get objective/fitness values
        obj_vals = [problem(x) for x in sample]

        # artificially resize to [-5, 5] while obj vals are calculated for the original values
        sample_resized = (sample - sample.min(axis=0)) / sample.ptp(axis=0) * (upper_bound - lower_bound) + lower_bound

        # calculate ELA features, if an exception occurs, impute the corresponding set with None values
        # (this usually happens only for ela_distr in combination with CMA-ES sampling where the Cov matrix degenerates to a very very small area)
        feat_object = create_feature_object(sample_resized, obj_vals, lower=lower_bound, upper=upper_bound)

        # Store information, which we want to write to the log file, within a python dict (ease of use)
        landscape_features = {
            **state_vars,
            **_calculate_feature_set(feat_object, 'ela_distr'),
            **_calculate_feature_set(feat_object, 'ela_meta'),
            **_calculate_feature_set(feat_object, 'nbc'),
            **_calculate_feature_set(feat_object, 'ic'),
            **_calculate_feature_set(feat_object, 'disp')
        }

        return landscape_features







