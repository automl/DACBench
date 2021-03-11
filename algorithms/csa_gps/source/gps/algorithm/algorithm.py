""" This file defines the BADMM-based GPS algorithm. """
import copy
import logging

import numpy as np
import scipy as sp

from gps.algorithm.algorithm_utils import PolicyInfo
from gps.algorithm.config import ALG
from gps.sample.sample_list import SampleList
from gps.algorithm.algorithm_utils import IterationData, TrajectoryInfo
from gps.utility.general_utils import extract_condition
from gps.proto.gps_pb2 import CUR_LOC, PAST_OBJ_VAL_DELTAS, CUR_SIGMA, CUR_PS, PAST_LOC_DELTAS, ACTION

LOGGER = logging.getLogger(__name__)


class Algorithm(object):
    """
    Sample-based joint policy learning and trajectory optimization with
    BADMM-based guided policy search algorithm.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(ALG)
        config.update(hyperparams)
        self._hyperparams = config

        if 'train_conditions' in hyperparams:
            self._cond_idx = hyperparams['train_conditions']
            self.M = len(self._cond_idx)
        else:
            self.M = hyperparams['conditions']
            self._cond_idx = range(self.M)
            self._hyperparams['train_conditions'] = self._cond_idx
            self._hyperparams['test_conditions'] = self._cond_idx
        self.iteration_count = 0

        # Grab a few values from the agent.
        agent = self._hyperparams['agent']
        self.agent = agent
        
        self.T = self._hyperparams['T'] = agent.T
        self.dU = self._hyperparams['dU'] = agent.dU
        self.dX = self._hyperparams['dX'] = agent.dX
        self.dO = self._hyperparams['dO'] = agent.dO

        init_traj_distr = config['init_traj_distr']
        init_traj_distr['x0'] = agent.x0
        init_traj_distr['dX'] = agent.dX
        init_traj_distr['dU'] = agent.dU
        del self._hyperparams['agent']  # Don't want to pickle this.

        # IterationData objects for each condition.
        self.cur = [IterationData() for _ in range(self.M)]
        self.prev = [IterationData() for _ in range(self.M)]

        dynamics = self._hyperparams['dynamics']
        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()
            self.cur[m].traj_info.dynamics = dynamics['type'](dynamics)
            cur_init_traj_distr = extract_condition(
                init_traj_distr, self._cond_idx[m]
            )
            cur_init_traj_distr['cur_cond_idx'] = self._cond_idx[m]
            self.cur[m].traj_distr = cur_init_traj_distr['type'](cur_init_traj_distr, agent)
        
        self.traj_opt = hyperparams['traj_opt']['type'](
            hyperparams['traj_opt']
        )
        self.cost = []
        for m in range(self.M):
            cost_hyperparams = hyperparams['cost'].copy()
            cost_hyperparams['cur_cond_idx'] = self._cond_idx[m]
            self.cost.append(hyperparams['cost']['type'](cost_hyperparams))
        
        self.base_kl_step = self._hyperparams['kl_step']
        
        policy_prior = self._hyperparams['policy_prior']
        for m in range(self.M):
            self.cur[m].pol_info = PolicyInfo(self._hyperparams)
            self.cur[m].pol_info.policy_prior = \
                    policy_prior['type'](policy_prior)

        self.policy_opt = self._hyperparams['policy_opt']['type'](
            self._hyperparams['policy_opt'], self.dO, self.dU
        )
    
    # policies is a list of M policies
    def print_policy_cost(self, policies, num_samples = 5):
        for m in range(self.M):
            all_cs = np.empty((num_samples, self.T))
            for i in range(num_samples):
                sample = self.agent.sample(policies[m], self._cond_idx[m], save=False)
                # cs has shape of (T,)
                cs = self.cost[m].eval(sample,True)[0]
                all_cs[i,:] = cs
            total_cs = np.sum(all_cs, axis=1)
            print("[Condition %d] Cumulative Costs: %s, Mean Cumulative Cost: %.4f" % (m,repr(total_cs.tolist()),np.mean(total_cs)))
    
    def iteration(self, sample_lists):
        """
        Run iteration of BADMM-based guided policy search.

        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        print("Algorithm train functions %d" % self.M)
        print("Sampled functions %d" % len(sample_lists))
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]

        if self.iteration_count == 0:
            print("Initial Trajectories")
            self.print_policy_cost([self.cur[m].traj_distr for m in range(self.M)])
            
        self._set_interp_values()
        self._update_dynamics()  # Update dynamics model using all sample.
        self._update_step_size()  # KL Divergence step size.

        for m in range(self.M):
            # save initial kl for debugging / visualization
            self.cur[m].pol_info.init_kl = self._policy_kl(m)[0]
        
        # Run inner loop to compute new policies.
        for inner_itr in range(self._hyperparams['inner_iterations']):
            #TODO: Could start from init controller.
            if self.iteration_count > 0 or inner_itr > 0:
                # Update the policy.
                self._update_policy(inner_itr)
            for m in range(self.M):
                self._update_policy_fit(m)  # Update policy priors.
            if self.iteration_count > 0 or inner_itr > 0:
                step = (inner_itr == self._hyperparams['inner_iterations'] - 1)
                # Update dual variables.
                for m in range(self.M):
                    self._policy_dual_step(m, step=step)
            self._update_trajectories()
            
            print("New Trajectories")
            self.print_policy_cost(self.new_traj_distr)

        self._advance_iteration_variables()

    def _set_interp_values(self):
        """
        Use iteration-based interpolation to set values of some
        schedule-based parameters.
        """
        # Compute temporal interpolation value.
        t = min((self.iteration_count + 1.0) /
                (self._hyperparams['iterations'] - 1), 1)
        # Perform iteration-based interpolation of entropy penalty.
        if type(self._hyperparams['ent_reg_schedule']) in (int, float):
            self.policy_opt.set_ent_reg(self._hyperparams['ent_reg_schedule'])
        else:
            sch = self._hyperparams['ent_reg_schedule']
            self.policy_opt.set_ent_reg(
                np.exp(np.interp(t, np.linspace(0, 1, num=len(sch)),
                                 np.log(sch)))
            )
        # Perform iteration-based interpolation of Lagrange multiplier.
        if type(self._hyperparams['lg_step_schedule']) in (int, float):
            self._hyperparams['lg_step'] = self._hyperparams['lg_step_schedule']
        else:
            sch = self._hyperparams['lg_step_schedule']
            self._hyperparams['lg_step'] = np.exp(
                np.interp(t, np.linspace(0, 1, num=len(sch)), np.log(sch))
            )

    def _update_step_size(self):
        """ Evaluate costs on samples, and adjust the step size. """
        # Evaluate cost function for all conditions and samples.
        for m in range(self.M):
            self._update_policy_fit(m, init=True)
            self._eval_cost(m)
            # Adjust step size relative to the previous iteration.
            if self.iteration_count >= 1 and self.prev[m].sample_list:
                self._stepadjust(m)

    def _update_policy(self, inner_itr):
        """ Compute the new policy. """
        dU, dO, T = self.dU, self.dO, self.T
        # Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, T, dO)), np.zeros((0, T, dU))
        tgt_prc, tgt_wt = np.zeros((0, T, dU, dU)), np.zeros((0, T))
        for m in range(self.M):
            samples = self.cur[m].sample_list
            X = samples.get_X()
            step = samples.get(CUR_SIGMA)
            N = len(samples)
            if inner_itr > 0:
                traj, pol_info = self.new_traj_distr[m], self.cur[m].pol_info
            else:
                traj, pol_info = self.cur[m].traj_distr, self.cur[m].pol_info
            mu = np.zeros((N, T, dU))
            prc = np.ones((N, T, dU, dU))
            wt = np.zeros((N, T))
            # Get time-indexed actions.
            for t in range(T):
                # Compute actions along this trajectory.
                for i in range(N):
                    mu[i, t, :] = \
                            (step[i, t, :])
                wt[:, t].fill(pol_info.pol_wt[t])
            tgt_mu = np.concatenate((tgt_mu, mu))
            tgt_prc = np.concatenate((tgt_prc, prc))
            tgt_wt = np.concatenate((tgt_wt, wt))
            obs_data = np.concatenate((obs_data, samples.get_obs()))
        self.policy_opt.update(obs_data, tgt_mu, tgt_prc, tgt_wt)
    
    # Fit linear model for mean policy action given state
    def _update_policy_fit(self, m, init=False):
        """
        Re-estimate the local policy values in the neighborhood of the
        trajectory.
        Args:
            m: Condition
            init: Whether this is the initial fitting of the policy.
        """
        print(m)
        print(self.cur[m].sample_list)
        dX, dU, T = self.dX, self.dU, self.T
        # Choose samples to use.
        samples = self.cur[m].sample_list
        N = len(samples)
        pol_info = self.cur[m].pol_info
        X = samples.get_X()
        obs = samples.get_obs()
        pol_mu, pol_sig = self.policy_opt.prob(obs)[:2]
        
        # Update policy prior.
        policy_prior = pol_info.policy_prior
        if init:
            samples = SampleList(self.cur[m].sample_list)
            mode = self._hyperparams['policy_sample_mode']
        else:
            samples = SampleList([])
            mode = 'add' # Don't replace with empty samples
        policy_prior.update(samples, self.policy_opt, mode)

        # Fit linearization and store in pol_info.
        pol_info.pol_K, pol_info.pol_k, pol_info.pol_S = \
                policy_prior.fit(X, pol_mu, pol_sig)
        for t in range(T):
            pol_info.chol_pol_S[t, :, :] = \
                    sp.linalg.cholesky(pol_info.pol_S[t, :, :])

    def _policy_dual_step(self, m, step=False):
        """
        Update the dual variables for the specified condition.
        Args:
            m: Condition
            step: Whether or not to update pol_wt.
        """
        dU, T = self.dU, self.T
        samples = self.cur[m].sample_list
        N = len(samples)
        X = samples.get_X()
        step1 = samples.get(CUR_SIGMA)
        if 'new_traj_distr' in dir(self):
            traj, pol_info = self.new_traj_distr[m], self.cur[m].pol_info
        else:
            traj, pol_info = self.cur[m].traj_distr, self.cur[m].pol_info

        # Compute trajectory action at each sampled state.
        traj_mu = np.zeros((N, T, dU))
        for i in range(N):
            for t in range(T):
                traj_mu[i, t, :] = step1[i, t, :]
        
        obs = samples.get_obs()
        pol_mu = self.policy_opt.prob(obs, True)[0]
        
        # Compute the difference and increment based on pol_wt.
        for t in range(T):
            tU, pU = traj_mu[:, t, :], pol_mu[:, t, :]
            # Increment mean term.
            pol_info.lambda_k[t, :] -= self._hyperparams['policy_dual_rate'] * \
                    pol_info.pol_wt[t] * \
                    traj.inv_pol_covar[t, :, :].dot(np.mean(tU - pU, axis=0))
            # Increment covariance term.
            t_covar, p_covar = traj.K[t, :, :], pol_info.pol_K[t, :, :]
            pol_info.lambda_K[t, :, :] -= \
                    self._hyperparams['policy_dual_rate_covar'] * \
                    pol_info.pol_wt[t] * \
                    traj.inv_pol_covar[t, :, :].dot(t_covar - p_covar)
        # Compute KL divergence.
        kl_m = self._policy_kl(m)[0]
        if step:
            lg_step = self._hyperparams['lg_step']
            # Increment pol_wt based on change in KL divergence.
            if self._hyperparams['fixed_lg_step'] == 1:
                # Take fixed size step.
                pol_info.pol_wt = np.array([
                    max(wt + lg_step, 0) for wt in pol_info.pol_wt
                ])
            elif self._hyperparams['fixed_lg_step'] == 2:
                # (In/De)crease based on change in constraint
                # satisfaction.
                if hasattr(pol_info, 'prev_kl'):
                    kl_change = kl_m / pol_info.prev_kl
                    for i in range(len(pol_info.pol_wt)):
                        if kl_change[i] < 0.8:
                            pol_info.pol_wt[i] *= 0.5
                        elif kl_change[i] >= 0.95:
                            pol_info.pol_wt[i] *= 2.0
            elif self._hyperparams['fixed_lg_step'] == 3:
                # (In/De)crease based on difference from average.
                if hasattr(pol_info, 'prev_kl'):
                    lower = np.mean(kl_m) - \
                            self._hyperparams['exp_step_lower'] * np.std(kl_m)
                    upper = np.mean(kl_m) + \
                            self._hyperparams['exp_step_upper'] * np.std(kl_m)
                    for i in range(len(pol_info.pol_wt)):
                        if kl_m[i] < lower:
                            pol_info.pol_wt[i] *= \
                                    self._hyperparams['exp_step_decrease']
                        elif kl_m[i] >= upper:
                            pol_info.pol_wt[i] *= \
                                    self._hyperparams['exp_step_increase']
            else:
                # Standard DGD step.
                pol_info.pol_wt = np.array([
                    max(pol_info.pol_wt[t] + lg_step * kl_m[t], 0)
                    for t in range(T)
                ])
            pol_info.prev_kl = kl_m
    
    def _update_dynamics(self):
        """
        Instantiate dynamics objects and update prior. Fit dynamics to
        current samples.
        """
        for m in range(self.M):
            cur_data = self.cur[m].sample_list
            X = cur_data.get_X()
            U = cur_data.get_U()

            # Update prior and fit dynamics.
            self.cur[m].traj_info.dynamics.update_prior(cur_data)
            self.cur[m].traj_info.dynamics.fit(X, U)

            # Fit x0mu/x0sigma.
            x0 = X[:, 0, :]
            x0mu = np.mean(x0, axis=0)
            self.cur[m].traj_info.x0mu = x0mu
            self.cur[m].traj_info.x0sigma = np.diag(
                np.maximum(np.var(x0, axis=0),
                           self._hyperparams['initial_state_var'])
            )

            prior = self.cur[m].traj_info.dynamics.get_prior()
            if prior:
                mu0, Phi, priorm, n0 = prior.initial_state()
                N = len(cur_data)
                self.cur[m].traj_info.x0sigma += \
                        Phi + (N*priorm) / (N+priorm) * \
                        np.outer(x0mu-mu0, x0mu-mu0) / (N+n0)

    def _update_trajectories(self):
        """
        Compute new linear Gaussian controllers.
        """
        if not hasattr(self, 'new_traj_distr'):
            self.new_traj_distr = [
                self.cur[cond].traj_distr for cond in range(self.M)
            ]
        for cond in range(self.M):
            self.new_traj_distr[cond], self.cur[cond].eta = \
                    self.traj_opt.update(cond, self)

    def _eval_cost(self, cond):
        """
        Evaluate costs for all samples for a condition.
        Args:
            cond: Condition to evaluate cost on.
        """
        # Constants.
        T, dX, dU = self.T, self.dX, self.dU
        N = len(self.cur[cond].sample_list)
        
        # Compute cost.
        cs = np.zeros((N, T))
        cc = np.zeros((N, T))
        cv = np.zeros((N, T, dX+dU))
        Cm = np.zeros((N, T, dX+dU, dX+dU))
        for n in range(N):
            sample = self.cur[cond].sample_list[n]
            # Get costs.
            l, lx, lu, lxx, luu, lux = self.cost[cond].eval(sample)
            cc[n, :] = self.cost[cond].weight * dU * l
            cs[n, :] = l

            # Assemble matrix and vector.
            cv[n, :, :] = self.cost[cond].weight * dU * np.c_[lx, lu]
            Cm[n, :, :, :] = self.cost[cond].weight * dU * np.concatenate(
                (np.c_[lxx, np.transpose(lux, [0, 2, 1])], np.c_[lux, luu]),
                axis=1
            )

            # Adjust for expanding cost around a sample.
            X = sample.get_X()
            U = sample.get_U()
            yhat = np.c_[X, U]
            rdiff = -yhat
            rdiff_expand = np.expand_dims(rdiff, axis=2)
            cv_update = np.sum(Cm[n, :, :, :] * rdiff_expand, axis=1)
            cc[n, :] += np.sum(rdiff * cv[n, :, :], axis=1)+ 0.5 * \
                    np.sum(rdiff * cv_update, axis=1)

            cv[n, :, :] += cv_update

        # Fill in cost estimate.
        self.cur[cond].traj_info.cc = np.mean(cc, 0)  # Constant term (scalar).
        self.cur[cond].traj_info.cv = np.mean(cv, 0)  # Linear term (vector).
        self.cur[cond].traj_info.Cm = np.mean(Cm, 0)  # Quadratic term (matrix).

        self.cur[cond].cs = cs  # True value of cost.
    
    def _advance_iteration_variables(self):
        """
        Move all 'cur' variables to 'prev', reinitialize 'cur'
        variables, and advance iteration counter.
        """
        self.iteration_count += 1
        self.prev = self.cur
        # TODO: change IterationData to reflect new stuff better
        for m in range(self.M):
            self.prev[m].new_traj_distr = self.new_traj_distr[m]
        self.cur = [IterationData() for _ in range(self.M)]
        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()
            
            cur_dynamics_prior = self.prev[m].traj_info.dynamics.prior
            self.prev[m].traj_info.dynamics.prior = None
            self.cur[m].traj_info.dynamics = copy.deepcopy(self.prev[m].traj_info.dynamics)
            self.cur[m].traj_info.dynamics.prior = cur_dynamics_prior
            
            self.cur[m].step_mult = self.prev[m].step_mult
            self.cur[m].eta = self.prev[m].eta
            self.cur[m].traj_distr = self.new_traj_distr[m]
            
        delattr(self, 'new_traj_distr')
        
        for m in range(self.M):
            self.cur[m].traj_info.last_kl_step = \
                    self.prev[m].traj_info.last_kl_step
            
            cur_policy_prior = self.prev[m].pol_info.policy_prior
            self.prev[m].pol_info.policy_prior = None
            self.cur[m].pol_info = copy.deepcopy(self.prev[m].pol_info)
            self.cur[m].pol_info.policy_prior = cur_policy_prior
            
    def _stepadjust(self, m):
        """
        Calculate new step sizes.
        Args:
            m: Condition
        """

        # Compute values under Laplace approximation. This is the policy
        # that the previous samples were actually drawn from under the
        # dynamics that were estimated from the previous samples.
        prev_laplace_obj, prev_laplace_kl = self._estimate_cost(
            self.prev[m].traj_distr, self.prev[m].traj_info, self.prev[m].pol_info, m
        )
        # This is the policy that we just used under the dynamics that
        # were estimated from the previous samples (so this is the cost
        # we thought we would have).
        new_pred_laplace_obj, new_pred_laplace_kl = self._estimate_cost(
            self.cur[m].traj_distr, self.prev[m].traj_info, self.prev[m].pol_info, m
        )

        # This is the actual cost we have under the current trajectory
        # based on the latest samples.
        new_actual_laplace_obj, new_actual_laplace_kl = self._estimate_cost(
            self.cur[m].traj_distr, self.cur[m].traj_info, self.cur[m].pol_info, m
        )

        # Measure the entropy of the current trajectory (for printout).
        ent = self._measure_ent(m)

        # Compute actual objective values based on the samples.
        prev_mc_obj = np.mean(np.sum(self.prev[m].cs, axis=1), axis=0)
        new_mc_obj = np.mean(np.sum(self.cur[m].cs, axis=1), axis=0)

        # Compute sample-based estimate of KL divergence between policy
        # and trajectories.
        new_mc_kl = self._policy_kl(m)[0]
        if self.iteration_count >= 1 and self.prev[m].sample_list:
            prev_mc_kl = self._policy_kl(m, prev=True)[0]
        else:
            prev_mc_kl = np.zeros_like(new_mc_kl)

        # Compute full policy KL divergence objective terms by applying
        # the Lagrange multipliers.
        pol_wt = self.cur[m].pol_info.pol_wt
        prev_laplace_kl_sum = np.sum(prev_laplace_kl * pol_wt)
        new_pred_laplace_kl_sum = np.sum(new_pred_laplace_kl * pol_wt)
        new_actual_laplace_kl_sum = np.sum(new_actual_laplace_kl * pol_wt)
        prev_mc_kl_sum = np.sum(prev_mc_kl * pol_wt)
        new_mc_kl_sum = np.sum(new_mc_kl * pol_wt)

        LOGGER.debug(
            'Trajectory step: ent: %f cost: %f -> %f KL: %f -> %f',
            ent, prev_mc_obj, new_mc_obj, prev_mc_kl_sum, new_mc_kl_sum
        )
        
        # Compute predicted and actual improvement.
        predicted_impr = np.sum(prev_laplace_obj) + prev_laplace_kl_sum - \
                np.sum(new_pred_laplace_obj) - new_pred_laplace_kl_sum
        actual_impr = np.sum(prev_laplace_obj) + prev_laplace_kl_sum - \
                np.sum(new_actual_laplace_obj) - new_actual_laplace_kl_sum

        # Print improvement details.
        LOGGER.debug('Previous cost: Laplace: %f MC: %f',
                     np.sum(prev_laplace_obj), prev_mc_obj)
        LOGGER.debug('Predicted new cost: Laplace: %f MC: %f',
                     np.sum(new_pred_laplace_obj), new_mc_obj)
        LOGGER.debug('Actual new cost: Laplace: %f MC: %f',
                     np.sum(new_actual_laplace_obj), new_mc_obj)
        LOGGER.debug('Previous KL: Laplace: %f MC: %f',
                     np.sum(prev_laplace_kl), np.sum(prev_mc_kl))
        LOGGER.debug('Predicted new KL: Laplace: %f MC: %f',
                     np.sum(new_pred_laplace_kl), np.sum(new_mc_kl))
        LOGGER.debug('Actual new KL: Laplace: %f MC: %f',
                     np.sum(new_actual_laplace_kl), np.sum(new_mc_kl))
        LOGGER.debug('Previous w KL: Laplace: %f MC: %f',
                     prev_laplace_kl_sum, prev_mc_kl_sum)
        LOGGER.debug('Predicted w new KL: Laplace: %f MC: %f',
                     new_pred_laplace_kl_sum, new_mc_kl_sum)
        LOGGER.debug('Actual w new KL: Laplace %f MC: %f',
                     new_actual_laplace_kl_sum, new_mc_kl_sum)
        LOGGER.debug('Predicted/actual improvement: %f / %f',
                     predicted_impr, actual_impr)

        # Compute actual KL step taken at last iteration.
        actual_step = self.cur[m].traj_info.last_kl_step / \
                (self._hyperparams['kl_step'] * self.T)
        if actual_step < self.cur[m].step_mult:
            self.cur[m].step_mult = max(actual_step,
                                        self._hyperparams['min_step_mult'])

        self._set_new_mult(predicted_impr, actual_impr, m)

    def _policy_kl(self, m, prev=False):
        """
        Monte-Carlo estimate of KL divergence between policy and
        trajectory.
        """
        dU, T = self.dU, self.T
        if prev:
            traj, pol_info = self.prev[m].traj_distr, self.cur[m].pol_info
            samples = self.prev[m].sample_list
        else:
            traj, pol_info = self.cur[m].traj_distr, self.cur[m].pol_info
            samples = self.cur[m].sample_list
        N = len(samples)
        X, obs = samples.get_X(), samples.get_obs()
        step = samples.get(CUR_SIGMA)
        #print('Sigmas: %s' % (step))
        kl, kl_m = np.zeros((N, T)), np.zeros(T)
        kl_l, kl_lm = np.zeros((N, T)), np.zeros(T)
        # Compute policy mean and covariance at each sample.
        pol_mu, _, pol_prec, pol_det_sigma = self.policy_opt.prob(obs)
        # Compute KL divergence.
        for t in range(T):
            # Compute trajectory action at sample.
            traj_mu = np.zeros((N, dU))
            for i in range(N):
                traj_mu[i, :] = step[i, t, :]
            diff = pol_mu[:, t, :] - traj_mu
            tr_pp_ct = pol_prec[:, t, :, :]
            k_ln_det_ct = 0.5 * dU
            ln_det_cp = np.log(pol_det_sigma[:, t])
            # IMPORTANT: Note that this assumes that pol_prec does not
            #            depend on state!!!!
            #            (Only the last term makes this assumption.)
            d_pp_d = np.sum(diff * (diff.dot(pol_prec[1, t, :, :])), axis=1)
            kl[:, t] = 0.5 * ln_det_cp + 0.5 * d_pp_d
            tr_pp_ct_m = np.mean(tr_pp_ct, axis=0)
            kl_m[t] = 0.5 * np.sum(np.sum(tr_pp_ct_m, axis=0), axis=0) - \
                    k_ln_det_ct + 0.5 * np.mean(ln_det_cp) + \
                    0.5 * np.mean(d_pp_d)
            # Compute trajectory action at sample with Lagrange
            # multiplier.
            traj_mu = np.zeros((N, dU))
            for i in range(N):
                traj_mu[i, :] = step[i, t, :]
            # Compute KL divergence with Lagrange multiplier.
            diff_l = pol_mu[:, t, :] - traj_mu
            d_pp_d_l = np.sum(diff_l * (diff_l.dot(pol_prec[1, t, :, :])),
                              axis=1)
            kl_l[:, t] = 0.5 * np.sum(np.sum(tr_pp_ct, axis=1), axis=1) - \
                    k_ln_det_ct + 0.5 * ln_det_cp + 0.5 * d_pp_d_l
            kl_lm[t] = 0.5 * np.sum(np.sum(tr_pp_ct_m, axis=0), axis=0) - \
                    k_ln_det_ct + 0.5 * np.mean(ln_det_cp) + \
                    0.5 * np.mean(d_pp_d_l)
        return kl_m, kl, kl_lm, kl_l

    def _estimate_cost(self, traj_distr, traj_info, pol_info, m):
        """
        Compute Laplace approximation to expected cost.
        Args:
            traj_distr: A linear Gaussian policy object.
            traj_info: A TrajectoryInfo object.
            pol_info: Policy linearization info.
            m: Condition number.
        """
        # Constants.
        T, dU, dX = self.T, self.dU, self.dX

        # Perform forward pass (note that we repeat this here, because
        # traj_info may have different dynamics from the ones that were
        # used to compute the distribution already saved in traj).
        mu, sigma = self.traj_opt.forward(traj_distr, traj_info)

        # Compute cost.
        predicted_cost = np.zeros(T)
        for t in range(T):
            predicted_cost[t] = traj_info.cc[t] + 0.5 * \
                    (np.sum(sigma[t, :, :] * traj_info.Cm[t, :, :]) +
                     mu[t, :].T.dot(traj_info.Cm[t, :, :]).dot(mu[t, :])) + \
                    mu[t, :].T.dot(traj_info.cv[t, :])

        # Compute KL divergence.
        predicted_kl = np.zeros(T)
        for t in range(T):
            inv_pS = np.linalg.solve(
                pol_info.chol_pol_S[t, :, :],
                np.linalg.solve(pol_info.chol_pol_S[t, :, :].T, np.eye(dU))
            )
            Ufb = mu[t, :dX]
            diff = mu[t, dX:] - Ufb
            predicted_kl[t] = 0.5 * np.sum(traj_distr.pol_covar[t, :, :] * inv_pS) +\
                    np.sum(
                        np.log(np.diag(pol_info.chol_pol_S[t, :, :]))
                    ) - np.sum(
                        np.log(np.diag(traj_distr.chol_pol_covar[t, :, :]))
                    ) + 0.5 * dU

        return predicted_cost, predicted_kl

    def compute_costs(self, m, eta):
        """ Compute cost estimates used in the LQR backward pass. """
        traj_info, traj_distr = self.cur[m].traj_info, self.cur[m].traj_distr
        pol_info = self.cur[m].pol_info
        T, dU, dX = traj_distr.T, traj_distr.dU, traj_distr.dX
        Cm, cv = np.copy(traj_info.Cm), np.copy(traj_info.cv)

        # Modify policy action via Lagrange multiplier.
        cv[:, dX:] -= pol_info.lambda_k
        Cm[:, dX:, :dX] -= pol_info.lambda_K
        Cm[:, :dX, dX:] -= np.transpose(pol_info.lambda_K, [0, 2, 1])

        #Pre-process the costs with KL-divergence terms.
        TKLm = np.zeros((T, dX+dU, dX+dU))
        TKLv = np.zeros((T, dX+dU))
        PKLm = np.zeros((T, dX+dU, dX+dU))
        PKLv = np.zeros((T, dX+dU))
        fCm, fcv = np.zeros(Cm.shape), np.zeros(cv.shape)
        for t in range(T):
            K, k = np.ones((dU, dX)), np.ones((dU))
            inv_pol_covar = traj_distr.inv_pol_covar[t, :, :]
            # Trajectory KL-divergence terms.
            TKLm[t, :, :] = np.vstack([
                np.hstack([
                    K.T.dot(inv_pol_covar).dot(K),
                    -K.T.dot(inv_pol_covar)]),
                np.hstack([-inv_pol_covar.dot(K), inv_pol_covar])
            ])
            TKLv[t, :] = np.concatenate([
                K.T.dot(inv_pol_covar).dot(k), -inv_pol_covar.dot(k)
            ])
            # Policy KL-divergence terms.
            inv_pol_S = np.linalg.solve(
                pol_info.chol_pol_S[t, :, :],
                np.linalg.solve(pol_info.chol_pol_S[t, :, :].T, np.eye(dU))
            )
            KB, kB = np.ones((dU, dX)), np.ones((dU))
            PKLm[t, :, :] = np.vstack([
                np.hstack([KB.T.dot(inv_pol_S).dot(KB), -KB.T.dot(inv_pol_S)]),
                np.hstack([-inv_pol_S.dot(KB), inv_pol_S])
            ])
            PKLv[t, :] = np.concatenate([
                KB.T.dot(inv_pol_S).dot(kB), -inv_pol_S.dot(kB)
            ])
            wt = pol_info.pol_wt[t]
            fCm[t, :, :] = (Cm[t, :, :] + TKLm[t, :, :] * eta +
                            PKLm[t, :, :] * wt) / (eta + wt)
            fcv[t, :] = (cv[t, :] + TKLv[t, :] * eta +
                         PKLv[t, :] * wt) / (eta + wt)

        return fCm, fcv
    
    def _set_new_mult(self, predicted_impr, actual_impr, m):
        """
        Adjust step size multiplier according to the predicted versus
        actual improvement.
        """
        # Model improvement as I = predicted_dI * KL + penalty * KL^2,
        # where predicted_dI = pred/KL and penalty = (act-pred)/(KL^2).
        # Optimize I w.r.t. KL: 0 = predicted_dI + 2 * penalty * KL =>
        # KL' = (-predicted_dI)/(2*penalty) = (pred/2*(pred-act)) * KL.
        # Therefore, the new multiplier is given by pred/2*(pred-act).
        new_mult = predicted_impr / (2.0 * max(1e-4,
                                               predicted_impr - actual_impr))
        new_mult = max(0.1, min(5.0, new_mult))
        new_step = max(
            min(new_mult * self.cur[m].step_mult,
                self._hyperparams['max_step_mult']),
            self._hyperparams['min_step_mult']
        )
        self.cur[m].step_mult = new_step

        if new_mult > 1:
            LOGGER.debug('Increasing step size multiplier to %f', new_step)
        else:
            LOGGER.debug('Decreasing step size multiplier to %f', new_step)

    def _measure_ent(self, m):
        """ Measure the entropy of the current trajectory. """
        ent = 0
        for t in range(self.T):
            ent = ent + np.sum(
                np.log(np.diag(self.cur[m].traj_distr.chol_pol_covar[t, :, :]))
            )
        return ent
    
    def __getstate__(self):
        return {k: v for k, v in self.__dict__.iteritems() if (k != "_hyperparams" and k != "agent")}

