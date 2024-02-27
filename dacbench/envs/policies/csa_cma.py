"""Optimal policy for csa cma."""
from __future__ import annotations


def csa(env, state):
    """Get the optimal action."""
    u = env.es.sigma
    hsig = env.es.adapt_sigma.hsig(env.es)
    env.es.hsig = hsig
    delta = env.es.adapt_sigma.update2(env.es, function_values=env.cur_obj_val)
    u *= delta
    return u
