"""Optimal policy for luby."""
from __future__ import annotations


def luby_gen(i):
    """Generator for the Luby Sequence."""
    for k in range(1, 33):
        if i == ((1 << k) - 1):
            yield 1 << (k - 1)

    for k in range(1, 9999):
        if 1 << (k - 1) <= i < (1 << k) - 1:
            yield from luby_gen(i - (1 << (k - 1)) + 1)


def get_optimum(env, state):
    """Get the optimal action."""
    return env._next_goal
