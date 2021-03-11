""" This file defines linear regression with an arbitrary prior. """
import numpy as np

from gps.algorithm.algorithm_utils import gauss_fit_joint_prior

class DynamicsLRPrior(object):
    """ Dynamics with linear regression, with arbitrary prior. """
    def __init__(self, hyperparams):
        self._hyperparams = hyperparams
        
        # Fitted dynamics: x_t+1 = Fm * [x_t;u_t] + fv.
        self.Fm = np.array(np.nan)
        self.fv = np.array(np.nan)
        self.dyn_covar = np.array(np.nan)  # Covariance.
        
        self.Fm = None
        self.fv = None
        self.dyn_covar = None
        self.prior = \
                self._hyperparams['prior']['type'](self._hyperparams['prior'])
    
    def update_prior(self, samples):
        """ Update dynamics prior. """
        X = samples.get_X()
        U = samples.get_U()
        self.prior.update(X, U)

    def get_prior(self):
        """ Return the dynamics prior. """
        return self.prior

    def fit(self, X, U):
        """ Fit dynamics. """
        N, T, dX = X.shape
        dU = U.shape[2]

        if N == 1:
            raise ValueError("Cannot fit dynamics on 1 sample")

        self.Fm = np.zeros([T, dX, dX+dU])
        self.fv = np.zeros([T, dX])
        self.dyn_covar = np.zeros([T, dX, dX])

        it = slice(dX+dU)
        ip = slice(dX+dU, dX+dU+dX)
        # Fit dynamics with least squares regression.
        dwts = (1.0 / N) * np.ones(N)
        for t in range(T - 1):
            Ys = np.c_[X[:, t, :], U[:, t, :], X[:, t+1, :]]
            # Obtain Normal-inverse-Wishart prior.
            mu0, Phi, mm, n0 = self.prior.eval(dX, dU, Ys)
            sig_reg = np.zeros((dX+dU+dX, dX+dU+dX))
            sig_reg[it, it] = self._hyperparams['regularization']*np.eye(dX+dU)
            Fm, fv, dyn_covar = gauss_fit_joint_prior(Ys,
                        mu0, Phi, mm, n0, dwts, dX+dU, dX, sig_reg, self._hyperparams['clipping_thresh'])
            self.Fm[t, :, :] = Fm
            self.fv[t, :] = fv
            # Fm * [x; u] + fv gives the predicted state
            self.dyn_covar[t, :, :] = dyn_covar
        return self.Fm, self.fv, self.dyn_covar
    
    def copy(self):
        """ Return a copy of the dynamics estimate. """
        dyn = type(self)(self._hyperparams)
        dyn.Fm = np.copy(self.Fm)
        dyn.fv = np.copy(self.fv)
        dyn.dyn_covar = np.copy(self.dyn_covar)
        return dyn
