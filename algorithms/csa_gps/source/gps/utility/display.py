import numpy as np
import json
from gps.proto.gps_pb2 import CUR_LOC, ACTION, CUR_PS, CUR_SIGMA, PAST_SIGMA, PAST_OBJ_VAL_DELTAS, PAST_LOC_DELTAS
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

from matplotlib import rcParams
rcParams["font.size"] = "30"
rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
rcParams['figure.figsize'] = (16.0, 9.0)
rcParams['figure.frameon'] = True
rcParams['figure.edgecolor'] = 'k'
rcParams['grid.color'] = 'k'
rcParams['grid.linestyle'] = ':'
rcParams['grid.linewidth'] = 0.5
rcParams['axes.linewidth'] = 1
rcParams['axes.edgecolor'] = 'k'
rcParams['axes.grid.which'] = 'both'
rcParams['legend.frameon'] = 'True'
rcParams['legend.framealpha'] = 1

rcParams['ytick.major.size'] = 12
rcParams['ytick.major.width'] = 1.5
rcParams['ytick.minor.size'] = 6
rcParams['ytick.minor.width'] = 1
rcParams['xtick.major.size'] = 12
rcParams['xtick.major.width'] = 1.5
rcParams['xtick.minor.size'] = 6
rcParams['xtick.minor.width'] = 1

from datetime import datetime
class Display(object):

    def __init__(self, hyperparams):
        self._hyperparams = hyperparams
        self._log_filename = self._hyperparams['log_filename']
        self._plot_filename = self._hyperparams['plot_filename']
        self._first_update = True

    def _output_column_titles(self, algorithm, policy_titles=False):
        """
        Setup iteration data column titles: iteration, average cost, and for
        each condition the mean cost over samples, step size, linear Guassian
        controller entropies, and initial/final KL divergences for BADMM.
        """
        condition_titles = '%3s | %8s %12s' % ('', '', '')
        itr_data_fields  = '%3s | %8s %12s' % ('itr', 'avg_cost', 'avg_pol_cost')
        for m in range(algorithm.M):
            condition_titles += ' | %8s %9s %-7d' % ('', 'condition', m)
            itr_data_fields  += ' | %8s %8s %8s' % ('  cost  ', '  step  ', 'entropy ')
            condition_titles += ' %8s %8s %8s' % ('', '', '')
            itr_data_fields  += ' %8s %8s %8s %s ' % ('pol_cost', 'kl_div_i', 'kl_div_f', 'samples')
        self.append_output_text(condition_titles)
        self.append_output_text(itr_data_fields)

    def eval(self, sample, cur_cond_idx):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample:  A single sample
        """
        T = sample.T
        Du = sample.dU
        Dx = sample.dX

        cur_fcn = sample.agent.fcns[cur_cond_idx]['fcn_obj']

        final_l = np.zeros(T)

        x = sample.get(CUR_LOC)
        sigma_ = sample.get(CUR_SIGMA)
        sigma = [sigma_[i][0] for i in range(sigma_.shape[0])]
        _, dim = x.shape


        for t in range(T):
            final_l[t] = sample.trajectory[t]

        return x, sigma, final_l

    def get_sample_data(self, sample, cur_cond_idx):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample:  A single sample
        """
        T = sample.T
        Du = sample.dU
        Dx = sample.dX

        cur_fcn = sample.agent.fcns[cur_cond_idx]['fcn_obj']

        ps_ = sample.get(CUR_PS)
        ps = [ps_[i][0] for i in range(ps_.shape[0])]
        past_sigma = sample.get(PAST_SIGMA)
        past_obj_val_deltas = sample.get(PAST_OBJ_VAL_DELTAS)
        past_loc_deltas = sample.get(PAST_LOC_DELTAS)

        return ps, past_sigma, past_obj_val_deltas, past_loc_deltas

    def _update_iteration_data(self, algorithm, test_idx, test_fcns, pol_sample_lists, traj_sample_lists):
        """
        Update iteration data information: iteration, average cost, and for
        each condition the mean cost over samples, step size, linear Guassian
        controller entropies, and initial/final KL divergences for BADMM.
        """
        #data = {}
        #if pol_sample_lists is not None:
        #    pol_costs = [[np.sum(algorithm.cost[m].eval(pol_sample_lists[m][i],True)[0]) for i in range(len(pol_sample_lists[m]))]
        #             for m in range(len(cond_idx_list))]
        #if traj_sample_lists is not None:
        #    traj_costs = [[np.sum(algorithm.cost[m].eval(traj_sample_lists[m][i],True)[0]) for i in range(len(traj_sample_lists[m]))]
        #             for m in range(len(cond_idx_list))]

            #data['avg_cost'] = np.mean(pol_costs)
            #itr_data = '%s : %12.2f' % ('avg_cost', np.mean(pol_costs))
            #self.append_output_text(itr_data)
        #else:
        #      pol_costs = None
        #    itr_data = '%3d | %8.2f' % (itr, avg_cost)
        for m,idx in enumerate(test_idx):
            samples = len(pol_sample_lists[m])
            sample = np.random.randint(samples)
            sample_ = 'Sample_' + str(sample)
            test_fcn = test_fcns[m % len(test_fcns)]
            #itr_data = '%s%d' % ('Sample_', i)
            #self.append_output_text(itr_data)
            pol_avg_cost, pol_std, traj_avg_cost, traj_std, pol_avg_sigma, pol_sigma_std, traj_avg_sigma, traj_sigma_std, end_values = self.get_data(pol_sample_lists[m], traj_sample_lists[m], idx)
            self.plot_data(pol_sample_lists[m][0], traj_sample_lists[m][0], test_fcn, pol_avg_cost, traj_avg_cost, pol_avg_sigma, traj_avg_sigma, pol_std, traj_std, pol_sigma_std, traj_sigma_std, end_values)

            #data[function_str][sample_] = {'obj_values': list(obj_val)}
                #itr_data = '%s : %s ' % ('cur_loc', x)
                #self.append_output_text(itr_data)
                #itr_data = '%s : %s ' % ('obj_values', obj_val)
                #self.append_output_text(itr_data)
        #self.append_output_text(data)
        return pol_avg_cost

    def get_data(self, pol_samples, traj_samples, cur_cond):
        pol_avg_obj = []
        pol_avg_sigma = []
        traj_avg_obj = []
        traj_avg_sigma = []
        end_values = []
        for m in range(len(pol_samples)):
            _,p_sigma,p_obj_val = self.eval(pol_samples[m], cur_cond)
            _,t_sigma,t_obj_val = self.eval(traj_samples[m], cur_cond)
            pol_avg_obj.append(p_obj_val)
            pol_avg_sigma.append(p_sigma)
            traj_avg_obj.append(t_obj_val)
            traj_avg_sigma.append(t_sigma)
            end_values.append(p_obj_val[-1])
        return np.mean(pol_avg_obj, axis=0), np.std(pol_avg_obj, axis=0), np.mean(traj_avg_obj, axis=0), np.std(traj_avg_obj, axis=0), np.mean(pol_avg_sigma, axis=0), np.std(pol_avg_sigma, axis=0), np.mean(traj_avg_sigma, axis=0), np.std(traj_avg_sigma, axis=0), end_values

    def plot_data(self, pol_sample, traj_sample, cur_cond, pol_costs, traj_costs, pol_sigma, traj_sigma, pol_std, traj_std, pol_sigma_std, traj_sigma_std, end_values):
        #pol_ps, pol_past_sigma, pol_past_obj_val_deltas, pol_past_loc_deltas = self.get_sample_data(pol_sample,cur_cond)
        #traj_ps, traj_past_sigma, traj_past_obj_val_deltas, traj_past_loc_deltas = self.get_sample_data(traj_sample, cur_cond)
        log_text = {}
        log_text['Average costs LTO'] = list(pol_costs)
        log_text['Average costs CSA'] = list(traj_costs)
        log_text['End values LTO'] = list(end_values)
        log_text['Sigma LTO'] = list(pol_sigma)
        log_text['Sigma CSA'] = list(traj_sigma)
        log_text['Std costs LTO'] = list(pol_std)
        log_text['Std costs CSA'] = list(traj_std)
        log_text['Std Sigma LTO'] = list(pol_sigma_std)
        log_text['Std Sigma CSA'] = list(traj_sigma_std)

#        log_text += 'Ps LTO: %s \n' % (pol_ps)
#        log_text += 'Ps CSA: %s \n' % (traj_ps)
#        log_text += 'Past Sigma LTO: %s \n' % (pol_past_sigma)
#        log_text += 'Past Sigma CSA: %s \n' % (traj_past_sigma)
#        log_text += 'Past Obj Val Deltas LTO: %s \n' % (pol_past_obj_val_deltas)
#        log_text += 'Past Obj Val Deltas CSA: %s \n' % (traj_past_obj_val_deltas)
#        log_text += 'Past Loc Deltas LTO: %s \n' % (pol_past_loc_deltas)
#        log_text += 'Past Loc Deltas CSA: %s \n' % (traj_past_loc_deltas)
        self.append_output_text(log_text)
        methods = ["rlbbo", "csa"]
        labels = ["RLBBO", "CSA"]

        plt.tick_params(axis='x', which='minor')
        plt.legend(loc=0, fontsize=25, ncol=2)
        plt.title(cur_cond, fontsize=50)
        plt.xlabel("iteration", fontsize=50)
        plt.ylabel("objective value", fontsize=50)
        plt.fill_between(list(range(len(pol_costs))), np.subtract(pol_costs,pol_std), np.add(pol_costs,pol_std), color=sns.xkcd_rgb["medium green"], alpha=0.5)
        plt.plot(pol_costs,color=sns.xkcd_rgb["medium green"],
             linewidth=4, label='RLBBO')
        plt.fill_between(list(range(len(traj_costs))),np.subtract(traj_costs,traj_std), np.add(traj_costs,traj_std), color=sns.xkcd_rgb["denim blue"], alpha=0.5)
        plt.plot(traj_costs,color=sns.xkcd_rgb["denim blue"],
             linewidth=4, label='CSA')
        plt.legend()
        timestamp = datetime.now()
        time = str(timestamp)
        method = "Objective_value"
        plot_file = ('%s_%s_%s_%s.pdf' % (self._plot_filename, method, cur_cond, time))
        plt.savefig(plot_file, bbox_inches='tight')
        plt.show()
        plt.clf()

        plt.tick_params(axis='x', which='minor')
        plt.legend(loc=0, fontsize=25, ncol=2)
        plt.title(cur_cond, fontsize=50)
        plt.xlabel("iteration", fontsize=50)
        plt.ylabel("Step size", fontsize=50)
        plt.fill_between(list(range(len(pol_sigma))),np.subtract(pol_sigma,pol_sigma_std), np.add(pol_sigma,pol_sigma_std), color=sns.xkcd_rgb["medium green"], alpha=0.5)
        plt.plot(pol_sigma, color=sns.xkcd_rgb["medium green"],
             linewidth=4, label='RLBBO')
        plt.fill_between(list(range(len(traj_sigma))),np.subtract(traj_sigma,traj_sigma_std), np.add(traj_sigma,traj_sigma_std), color=sns.xkcd_rgb["denim blue"], alpha=0.5)
        plt.plot(traj_sigma,color=sns.xkcd_rgb["denim blue"],
             linewidth=4, label='CSA')
        plt.legend()
        timestamp = datetime.now()
        time = str(timestamp)
        method = "Step size"
        plot_file = ('%s_%s_%s_%s.pdf' % (self._plot_filename, method, cur_cond, time))
        plt.savefig(plot_file, bbox_inches='tight')
        plt.show()
        plt.clf()


    def update(self, algorithm, agent, test_fcns, cond_idx_list, pol_sample_lists, traj_sample_lists):

        if self._first_update:
            #self._output_column_titles(algorithm)
            self._first_update = False
        #costs = [np.mean(np.sum(algorithm.prev[m].cs, axis=1)) for m in range(algorithm.M)]
        pol_costs = self._update_iteration_data(algorithm, test_fcns, cond_idx_list, pol_sample_lists, traj_sample_lists)

        return pol_costs

    def append_output_text(self, text):
        with open(self._log_filename, 'a') as f:
            #f.write('%s \n' % (str(text)))
            json.dump(text, f)
        #print(text)

