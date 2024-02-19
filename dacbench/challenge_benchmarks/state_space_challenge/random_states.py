from __future__ import annotations

import numpy as np


def small_random_luby_state(self):
    core_state = self.get_default_state(None)
    num_random_elements = 20 - len(core_state)
    for _i in range(num_random_elements):
        core_state.append(np.random.normal(3, 2.5))
    return core_state


def random_luby_state(self):
    core_state = self.get_default_state(None)
    num_random_elements = 250 - len(core_state)
    for _i in range(num_random_elements):
        core_state.append(np.random.normal(3, 2.5))
    return core_state


def small_random_sigmoid_state(self):
    core_state = self.get_default_state(None)
    num_random_elements = 50 - len(core_state)
    state = []
    for _i in range(50):
        if num_random_elements > 0 and len(core_state) > 0:
            append_random = np.random.choice([0, 1])
            if append_random:
                state.append(np.random.normal(2, 1.5))
                num_random_elements -= 1
            else:
                state.append(core_state[0])
                core_state = core_state[1:]
        elif len(core_state) == 0:
            state.append(np.random.normal(2, 1.5))
        else:
            state.append(core_state[0])
            core_state = core_state[1:]
    return state


def random_sigmoid_state(self):
    core_state = self.get_default_state(None)
    num_random_elements = 500 - len(core_state)
    state = []
    for _i in range(500):
        if num_random_elements > 0 and len(core_state) > 0:
            append_random = np.random.choice([0, 1])
            if append_random:
                state.append(np.random.normal(2, 1.5))
                num_random_elements -= 1
            else:
                state.append(core_state[0])
                core_state = core_state[1:]
        elif len(core_state) == 0:
            state.append(np.random.normal(2, 1.5))
        else:
            state.append(core_state[0])
            core_state = core_state[1:]
    return state
