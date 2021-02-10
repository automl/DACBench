""" Default configuration and hyperparameter values for costs. """
import numpy as np

from gps.algorithm.cost.cost_utils import RAMP_CONSTANT

COST = {
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time.
    'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
    'weight': 1.0
}
