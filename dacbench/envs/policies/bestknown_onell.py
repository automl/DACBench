from dacbench.envs.onell_env import OneMax
import numpy as np


def get_dyn_theory(env, state):
    """
    return the best lambda known by theory (see [Doerr, Doerr, Ebel TCS 2015])
    only applicable when env.config.name is lbd_theory and env.config.problem is OneMax
    """
    assert env.problem == OneMax
    assert env.action_description == "lbd"
    assert env.state_description == "n, f(x)"
    assert len(env.state_functions) == 2

    return np.asarray([np.sqrt(env.x.n / (env.x.n - state[1]))], dtype=np.float32)


def get_dyn_onefifth(env, state):
    """
    return the best lambda using 1/5th success rule (see [Doerr, Doerr, Ebel TCS 2015])
    only applicable when env.config.name is lbd_onefifth and env.config.problem is OneMax
    """
    assert env.problem == OneMax
    assert env.action_description == "lbd"
    assert env.state_description == "n, delta f(x), lbd_{t-1}"
    assert len(env.state_functions) == 3

    delta_fx = state[1]
    lbd = state[2]

    n = env.x.n
    if delta_fx > 0:
        lbd = max(2 / 3 * lbd, 1)
    else:
        lbd = min(np.power(3 / 2, 1 / 4) * lbd, n - 1)
    # print(lbd)

    return np.asarray([lbd])
