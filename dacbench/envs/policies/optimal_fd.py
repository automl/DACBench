"""Optimal policy for Fast Downward."""
from __future__ import annotations

import json


def get_optimum(env, state):
    """Get the optimal action."""
    instance = env.get_instance()[:-12] + "optimal.json"
    with open(instance, "r+") as fp:
        optimal = json.load(fp)
    return optimal[env.c_step]
