"""Wrapper for instance sampling."""
from __future__ import annotations

import numpy as np
from gymnasium import Wrapper
from scipy.stats import norm


class InstanceSamplingWrapper(Wrapper):
    """Wrapper to sample a new instance at a given time point.

    Instances can either be sampled using a given method
    or a distribution infered from a given list of instances.
    """

    def __init__(self, env, sampling_function=None, instances=None, reset_interval=0):
        """Initialize wrapper.

        Either sampling_function or instances must be given

        Parameters
        ----------
        env : gym.Env
            Environment to wrap
        sampling_function : function
            Function to sample instances from
        instances : list
            List of instances to infer distribution from
        reset_interval : int
            additional episodes for which to keep an instance

        """
        super().__init__(env)
        if sampling_function:
            self.sampling_function = sampling_function
        elif instances:
            self.sampling_function = self.fit_dist(instances)
        else:
            raise Exception("No distribution to sample from given")
        self.reset_interval = reset_interval
        self.reset_tracker = 0

    def __setattr__(self, name, value):
        """Set attribute in wrapper if available and in env if not.

        Parameters
        ----------
        name : str
            Attribute to set
        value
            Value to set attribute to

        """
        if name in ["sampling_function", "env", "fit_dist", "reset"]:
            object.__setattr__(self, name, value)
        else:
            setattr(self.env, name, value)

    def __getattribute__(self, name):
        """Get attribute value of wrapper if available and of env if not.

        Parameters
        ----------
        name : str
            Attribute to get

        Returns:
        -------
        value
            Value of given name

        """
        if name in ["sampling_function", "env", "fit_dist", "reset"]:
            return object.__getattribute__(self, name)

        return getattr(self.env, name)

    def reset(self):
        """Reset environment and use sampled instance for training.

        Returns:
        -------
        np.array
            state

        """
        if self.reset_tracker >= self.reset_interval:
            instance = self.sampling_function()
            self.env.use_next_instance(instance=instance)
        return self.env.reset()

    def fit_dist(self, instances):
        """Approximate instance distribution in given instance set.

        Parameters
        ----------
        instances : List
            instance set

        Returns:
        -------
        method
            sampling method for new instances

        """
        dists = []
        for i in range(len(instances[0])):
            component = [instances[k][i] for k in instances]
            dist = norm.fit(component)
            dists.append(dist)

        def sample():
            instance = []
            rng = np.random.default_rng()
            for d in dists:
                instance.append(rng.normal(d[0], d[1]))
            return instance

        return sample
