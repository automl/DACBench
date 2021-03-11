""" This file defines the linear Gaussian policy class. """
import numpy as np
from cma.sigma_adaptation import CMAAdaptSigmaCSA
from gps.algorithm.policy.policy import Policy
from gps.utility.general_utils import check_shape


class ConstantPolicy(Policy):
    """
    Constant policy
    Important for RL learning ability check
    """
    def __init__(self, const=0.5):
        Policy.__init__(self)
        self.const = const
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
        if self.adapt_sigma is None:
            self.adapt_sigma = CMAAdaptSigmaCSA()
        self.adapt_sigma.sigma = es.sigma
        hsig = es.adapt_sigma.hsig(es)
        es.hsig = hsig
        es.adapt_sigma.update2(es, function_values=f_vals)
        u = self.const
        return u


