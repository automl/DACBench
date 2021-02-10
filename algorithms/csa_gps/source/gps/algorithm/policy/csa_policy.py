""" This file defines the linear Gaussian policy class. """
import numpy as np
from cma.sigma_adaptation import CMAAdaptSigmaCSA
from gps.algorithm.policy.policy import Policy


class CSAPolicy(Policy):
    """
    Time-varying linear Gaussian policy.
    U = CSA(sigma, ps, chiN)+ noise, where noise ~ N(0, chol_pol_covar)
    """

    def __init__(self, T=50):
        Policy.__init__(self)

        self.teacher = 0  # np.random.choice([0,1])
        self.T = T
        self.adapt_sigma = CMAAdaptSigmaCSA()

    def act(self, x, obs, t, noise, es, f_vals):
        """
        Return an action for a state.
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: Action noise. This will be scaled by the variance.
        """
        # if self.adapt_sigma is None:
        #    self.adapt_sigma = CMAAdaptSigmaCSA()

        # self.adapt_sigma.sigma = es.sigma
        u = es.sigma
        hsig = es.adapt_sigma.hsig(es)
        es.hsig = hsig
        # if self.teacher == 0 or t == 0 :
        delta = es.adapt_sigma.update2(es, function_values=f_vals)
        # else:
        #    delta = self.init_sigma
        u *= delta
        # if t == 0:
        #    self.init_sigma = delta
        return u
