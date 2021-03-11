""" This file defines utility classes and functions for costs. """
import numpy as np

RAMP_CONSTANT = 1
RAMP_LINEAR = 2
RAMP_QUADRATIC = 3
RAMP_FINAL_ONLY = 4
RAMP_CUSTOM = 5

def get_ramp_multiplier(ramp_option, T, wp_final_multiplier=1.0, wp_custom=None):
    """
    Return a time-varying multiplier.
    Returns:
        A (T,) float vector containing weights for each time step.
    """
    if ramp_option == RAMP_CONSTANT:
        wpm = np.ones(T)
    elif ramp_option == RAMP_LINEAR:
        wpm = (np.arange(T, dtype=np.float32) + 1) / T
    elif ramp_option == RAMP_QUADRATIC:
        wpm = ((np.arange(T, dtype=np.float32) + 1) / T) ** 2
    elif ramp_option == RAMP_FINAL_ONLY:
        wpm = np.zeros(T)
        wpm[T-1] = 1.0
    elif ramp_option == RAMP_CUSTOM:
        assert(wp_custom is not None)
        wpm = wp_custom
    else:
        raise ValueError('Unknown cost ramp requested!')
    wpm[-1] *= wp_final_multiplier
    return wpm

