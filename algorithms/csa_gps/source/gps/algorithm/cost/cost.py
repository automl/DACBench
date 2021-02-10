import copy

import numpy as np

from gps.algorithm.cost.config import COST
from gps.algorithm.cost.cost_utils import get_ramp_multiplier

from gps.proto.gps_pb2 import CUR_LOC


class Cost(object):
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST)
        config.update(hyperparams)
        self._hyperparams = config
        # Used by _eval_cost in algorithm.py
        self.weight = self._hyperparams["weight"]
        self.cur_cond_idx = self._hyperparams["cur_cond_idx"]

    def eval(self, sample, obj_val_only=False):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample:  A single sample
        """
        T = sample.T
        Du = sample.dU
        Dx = sample.dX

        # cur_fcn = sample.agent.fcns[self.cur_cond_idx]['fcn_obj']

        final_l = np.zeros(T)

        if not obj_val_only:
            final_lu = np.zeros((T, Du))
            final_lx = np.zeros((T, Dx))
            final_luu = np.zeros((T, Du, Du))
            final_lxx = np.zeros((T, Dx, Dx))
            final_lux = np.zeros((T, Du, Dx))

        x = sample.get("current_loc")
        _, dim = x.shape

        # Time step-specific weights
        wpm = get_ramp_multiplier(
            self._hyperparams["ramp_option"],
            T,
            wp_final_multiplier=self._hyperparams["wp_final_multiplier"],
            wp_custom=self._hyperparams["wp_custom"]
            if "wp_custom" in self._hyperparams
            else None,
        )

        if not obj_val_only:
            ls = np.empty((T, dim))
            lss = np.empty((T, dim, dim))

        # cur_fcn.new_sample(batch_size="all")    # Get noiseless gradient
        for t in range(T):
            final_l[t] = sample.trajectory[t]  # cur_fcn.evaluate(x[t,:])
            # if not obj_val_only:
            # ls[t,:] = cur_fcn.grad(x[t,:][:,None])[:,0]
            # lss[t,:,:] = cur_fcn.hess(x[t,:][:,None])

        final_l = final_l * wpm

        # if not obj_val_only:
        # ls = ls * wpm[:,None]
        # lss = lss * wpm[:,None,None]

        # Equivalent to final_lx[:,sensor_start_idx:sensor_end_idx] = ls
        # sample.agent.pack_data_x(final_lx, ls, data_types=[CUR_LOC])
        # Equivalent to final_lxx[:,sensor_start_idx:sensor_end_idx,sensor_start_idx:sensor_end_idx] = lss
        # sample.agent.pack_data_x(final_lxx, lss, data_types=[CUR_LOC, CUR_LOC])

        if obj_val_only:
            return (final_l,)
        else:
            return final_l, final_lx, final_lu, final_lxx, final_luu, final_lux
