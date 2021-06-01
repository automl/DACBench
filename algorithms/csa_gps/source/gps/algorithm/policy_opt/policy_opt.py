""" This file defines policy optimization for a tensorflow policy. """
import copy
import logging

import numpy as np

# NOTE: Order of these imports matters for some reason.
# Changing it can lead to segmentation faults on some machines.

from gps.algorithm.policy_opt.config import POLICY_OPT
import tensorflow as tf

from gps.algorithm.policy.tf_policy import TfPolicy
from gps.algorithm.policy_opt.tf_utils import TfSolver


LOGGER = logging.getLogger(__name__)


class PolicyOpt(object):
    """ Policy optimization using tensor flow for DAG computations/nonlinear function approximation. """

    def __init__(self, hyperparams, dO, dU):
        config = copy.deepcopy(POLICY_OPT)
        config.update(hyperparams)

        self._hyperparams = config
        self._dO = dO
        self._dU = dU

        tf.set_random_seed(self._hyperparams["random_seed"])

        self.tf_iter = 0
        self.batch_size = self._hyperparams["batch_size"]
        self.device_string = "/cpu:0"
        if self._hyperparams["use_gpu"] == 1:
            self.gpu_device = self._hyperparams["gpu_id"]
            self.device_string = "/gpu:" + str(self.gpu_device)
        self.act_op = None  # mu_hat
        self.loss_scalar = None
        self.obs_tensor = None
        self.precision_tensor = None
        self.action_tensor = None  # mu true
        self.solver = None
        self.init_network()
        self.init_solver()
        self.var = self._hyperparams["init_var"] * np.ones(dU)
        self.sess = tf.Session()
        self.policy = TfPolicy(
            dU,
            self.obs_tensor,
            self.act_op,
            np.zeros(dU),
            self.sess,
            self.device_string,
        )
        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)

    def init_network(self):
        """ Helper method to initialize the tf networks used """
        tf_map_generator = self._hyperparams["network_model"]
        tf_map = tf_map_generator(
            dim_input=self._dO,
            dim_output=self._dU,
            batch_size=self.batch_size,
            network_config=self._hyperparams["network_params"],
        )
        self.obs_tensor = tf_map.get_input_tensor()
        self.action_tensor = tf_map.get_target_output_tensor()
        self.precision_tensor = tf_map.get_precision_tensor()
        self.act_op = tf_map.get_output_op()
        self.loss_scalar = tf_map.get_loss_op()

    def init_solver(self):
        """ Helper method to initialize the solver. """
        self.solver = TfSolver(
            loss_scalar=self.loss_scalar,
            solver_name=self._hyperparams["solver_type"],
            base_lr=self._hyperparams["lr"],
            lr_policy=self._hyperparams["lr_policy"],
            momentum=self._hyperparams["momentum"],
            momentum2=self._hyperparams["momentum2"],
            epsilon=self._hyperparams["epsilon"],
            weight_decay=self._hyperparams["weight_decay"],
        )

    def update(self, obs, tgt_mu, tgt_prc, tgt_wt):
        """
        Update policy.
        Args:
            obs: Numpy array of observations, N x T x dO.
            tgt_mu: Numpy array of mean controller outputs, N x T x dU.
            tgt_prc: Numpy array of precision matrices, N x T x dU x dU.
            tgt_wt: Numpy array of weights, N x T.
        Returns:
            A tensorflow object with updated weights.
        """
        N, T = obs.shape[:2]
        dU, dO = self._dU, self._dO

        # Renormalize weights.
        tgt_wt *= float(N * T) / np.sum(tgt_wt)
        # Allow weights to be at most twice the robust median.
        mn = np.median(tgt_wt[(tgt_wt > 1e-2).nonzero()])
        for n in range(N):
            for t in range(T):
                tgt_wt[n, t] = min(tgt_wt[n, t], 2 * mn)
        # Robust median should be around one.
        tgt_wt /= mn

        # Reshape inputs.
        obs = np.reshape(obs, (N * T, dO))
        tgt_mu = np.reshape(tgt_mu, (N * T, dU))
        tgt_prc = np.reshape(tgt_prc, (N * T, dU, dU))
        tgt_wt = np.reshape(tgt_wt, (N * T, 1, 1))

        # Fold weights into tgt_prc.
        tgt_prc = tgt_wt * tgt_prc

        # Normalize obs, but only compute normalzation at the beginning.
        if self.policy.scale is None or self.policy.bias is None:
            # 1e-3 to avoid infs if some state dimensions don't change in the
            # first batch of samples
            self.policy.scale = np.diag(1.0 / np.maximum(np.std(obs, axis=0), 1e-3))
            self.policy.bias = -np.mean(obs.dot(self.policy.scale), axis=0)
        obs = obs.dot(self.policy.scale) + self.policy.bias

        # Assuming that N*T >= self.batch_size.
        batches_per_epoch = np.floor(N * T / self.batch_size)
        idx = list(range(N * T))
        average_loss = 0
        np.random.shuffle(idx)

        # actual training.
        for i in range(self._hyperparams["iterations"]):
            # Load in data for this batch.
            start_idx = int(i * self.batch_size % (batches_per_epoch * self.batch_size))
            idx_i = idx[start_idx : start_idx + self.batch_size]
            feed_dict = {
                self.obs_tensor: obs[idx_i],
                self.action_tensor: tgt_mu[idx_i],
                self.precision_tensor: tgt_prc[idx_i],
            }
            train_loss = self.solver(feed_dict, self.sess)

            average_loss += train_loss
            if (i + 1) % 500 == 0:
                LOGGER.debug(
                    "tensorflow iteration %d, average loss %f",
                    i + 1,
                    average_loss / 500,
                )
                print("supervised tf loss is " + str(average_loss))
                average_loss = 0

        # Keep track of tensorflow iterations for loading solver states.
        self.tf_iter += self._hyperparams["iterations"]

        # Optimize variance.

        self.var = (
            np.sum(tgt_wt, axis=0)[:, 0] - 2 * N * T * self._hyperparams["ent_reg"]
        ) / np.sum(np.diagonal(tgt_prc, axis1=1, axis2=2), axis=0)

        self.policy.chol_pol_covar = np.diag(np.sqrt(self.var))

        return self.policy

    def prob(self, obs, diag_var=False):
        """
        Run policy forward.
        Args:
            obs: Numpy array of observations that is N x T x dO.
        """
        dU = self._dU
        N, T = obs.shape[:2]

        output = np.zeros((N, T, dU))

        for i in range(N):
            for t in range(T):
                # Feed in data.
                if self.policy.scale is not None:
                    feed_dict = {
                        self.obs_tensor: np.expand_dims(obs[i, t], axis=0).dot(
                            self.policy.scale
                        )
                        + self.policy.bias
                    }
                else:
                    feed_dict = {self.obs_tensor: np.expand_dims(obs[i, t], axis=0)}
                with tf.device(self.device_string):
                    output[i, t, :] = self.sess.run(self.act_op, feed_dict=feed_dict)

        if diag_var:
            pol_sigma = np.tile(self.var, [N, T, 1])
            pol_prec = np.tile(1.0 / self.var, [N, T, 1])
            pol_det_sigma = np.tile(np.prod(self.var), [N, T])
        else:
            pol_sigma = np.tile(np.diag(self.var), [N, T, 1, 1])
            pol_prec = np.tile(np.diag(1.0 / self.var), [N, T, 1, 1])
            pol_det_sigma = np.tile(np.prod(self.var), [N, T])

        return output, pol_sigma, pol_prec, pol_det_sigma

    def set_ent_reg(self, ent_reg):
        """ Set the entropy regularization. """
        self._hyperparams["ent_reg"] = ent_reg

    # For pickling.
    def __getstate__(self):

        return {
            "hyperparams": self._hyperparams,
            "dO": self._dO,
            "dU": self._dU,
            "scale": self.policy.scale,
            "bias": self.policy.bias,
            "tf_iter": self.tf_iter,
        }

    # For unpickling.
    def __setstate__(self, state):
        self.__init__(state["hyperparams"], state["dO"], state["dU"])
        self.policy.scale = state["scale"]
        self.policy.bias = state["bias"]
        self.tf_iter = state["tf_iter"]
