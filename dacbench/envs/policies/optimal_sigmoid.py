import numpy as np


def sig(x, scaling, inflection):
    """ Simple sigmoid function """
    return 1 / (1 + np.exp(-scaling * (x - inflection)))


def get_optimum(env, state):
    sigmoids = [
        np.abs(sig(env.c_step, slope, shift))
        for slope, shift in zip(env.shifts, env.slopes)
    ]
    action = []
    for i in range(len(env.action_vals)):
        best_action = None
        dist = 100
        for a in range(env.action_vals[i] + 1):
            if np.abs(sigmoids[i] - a / (env.action_vals[i] - 1)) < dist:
                dist = np.abs(sigmoids[i] - a / (env.action_vals[i]))
                best_action = a
        action.append(best_action)
    for k in env.action_mapper.keys():
        if env.action_mapper[k] == tuple(action):
            return k
