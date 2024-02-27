"""Optimal policy for sigmoid."""
from __future__ import annotations

import numpy as np


def sig(x, scaling, inflection):
    """Simple sigmoid function."""
    return 1 / (1 + np.exp(-scaling * (x - inflection)))


def get_optimum(env, state):
    """Get the optimal action."""
    sigmoids = [
        np.abs(sig(env.c_step, slope, shift))
        for slope, shift in zip(env.shifts, env.slopes, strict=False)
    ]
    action = []
    for i in range(len(env.action_space.nvec)):
        best_action = None
        dist = 100
        for a in range(env.action_space.nvec[i] + 1):
            if np.abs(sigmoids[i] - a / (env.action_space.nvec[i])) < dist:
                dist = np.abs(sigmoids[i] - a / (env.action_space.nvec[i] + 1))
                best_action = a
        action.append(best_action)
    return action
