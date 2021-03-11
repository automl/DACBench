""" This file defines general utility functions and classes. """
import numpy as np

class BundleType(object):
    """
    This class bundles many fields, similar to a record or a mutable
    namedtuple.
    """
    def __init__(self, variables):
        for var, val in variables.items():
            object.__setattr__(self, var, val)

    # Freeze fields so new ones cannot be set.
    def __setattr__(self, key, value):
        if not hasattr(self, key):
            raise AttributeError("%r has no attribute %s" % (self, key))
        object.__setattr__(self, key, value)

def check_shape(value, expected_shape, name=''):
    """
    Throws a ValueError if value.shape != expected_shape.
    Args:
        value: Matrix to shape check.
        expected_shape: A tuple or list of integers.
        name: An optional name to add to the exception message.
    """
    if value.shape != tuple(expected_shape):
        raise ValueError('Shape mismatch %s: Expected %s, got %s' %
                         (name, str(expected_shape), str(value.shape)))


def finite_differences(func, inputs, func_output_shape=(), epsilon=1e-5):
    """
    Computes gradients via finite differences.
    derivative = (func(x+epsilon) - func(x-epsilon)) / (2*epsilon)
    Args:
        func: Function to compute gradient of. Inputs and outputs can be
            arbitrary dimension.
        inputs: Vector value to compute gradient at.
        func_output_shape: Shape of the output of func. Default is
            empty-tuple, which works for scalar-valued functions.
        epsilon: Difference to use for computing gradient.
    Returns:
        Gradient vector of each dimension of func with respect to each
        dimension of input.
    """
    gradient = np.zeros(inputs.shape+func_output_shape)
    for idx, _ in np.ndenumerate(inputs):
        test_input = np.copy(inputs)
        test_input[idx] += epsilon
        obj_d1 = func(test_input)
        assert obj_d1.shape == func_output_shape
        test_input = np.copy(inputs)
        test_input[idx] -= epsilon
        obj_d2 = func(test_input)
        assert obj_d2.shape == func_output_shape
        diff = (obj_d1 - obj_d2) / (2 * epsilon)
        gradient[idx] += diff
    return gradient


def approx_equal(a, b, threshold=1e-5):
    """
    Return whether two numbers are equal within an absolute threshold.
    Returns:
        True if a and b are equal within threshold.
    """
    return np.all(np.abs(a - b) < threshold)


def extract_condition(hyperparams, m):
    """
    Pull the relevant hyperparameters corresponding to the specified
    condition, and return a new hyperparameter dictionary.
    """
    return {var: val[m] if isinstance(val, list) else val
            for var, val in hyperparams.items()}

def replicate_var(val, num_conds):
    return val if isinstance(val, list) else [val] * num_conds
