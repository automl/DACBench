""" Initializations for linear Gaussian controllers. """
import copy
import numpy as np
from gps.algorithm.policy.config import INIT_LG
from gps.algorithm.policy.csa_policy import CSAPolicy
from gps.algorithm.policy.lin_gauss_policy import LinearGaussianPolicy
from dacbench.benchmarks import CMAESBenchmark
from gym import spaces


def init_cmaes_controller(hyperparams, agent):

    config = copy.deepcopy(INIT_LG)
    config.update(hyperparams)

    dX, dU = config["dX"], config["dU"]
    T = config["T"]
    cur_cond_idx = config["cur_cond_idx"]
    bench = CMAESBenchmark()
    bench.config.popsize = agent.popsize
    bench.config.hist_length = agent.history_len
    bench.config.observation_space_args = [
        {
            "current_loc": spaces.Box(
                low=-np.inf, high=np.inf, shape=np.arange(agent.input_dim).shape
            ),
            "past_deltas": spaces.Box(
                low=-np.inf, high=np.inf, shape=np.arange(bench.config.hist_length).shape
            ),
            "current_ps": spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            "current_sigma": spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            "history_deltas": spaces.Box(
                low=-np.inf, high=np.inf, shape=np.arange(bench.config.hist_length * 2).shape
            ),
            "past_sigma_deltas": spaces.Box(
                low=-np.inf, high=np.inf, shape=np.arange(bench.config.hist_length).shape
            ),
        }
    ]
    world = bench.get_environment()

    if config["verbose"]:
        print("Finding Initial Linear Gaussian Controller")
    action_mean = []
    action_var = []
    for i in range(25):
        f_values = []
        cur_policy = CSAPolicy(T=T)

        state = world.reset()
        for t in range(T):
            X_t = agent.get_vectorized_state(state, cur_cond_idx)
            es = world.es
            f_vals = world.func_values
            U_t = cur_policy.act(X_t, None, t, np.zeros((dU,)), es, f_vals)
            world.step(U_t)
            f_values.append(U_t)
        action_mean.append(f_values)  # np.mean(f_values, axis=0))
        action_var.append(f_values)  # np.mean(f_values, axis=0))
    mean_actions = np.mean(action_mean, axis=0)
    var_actions = np.std(action_var, axis=0)
    np.place(var_actions, var_actions == 0, config["init_var"])
    Kt = np.zeros((dU, dX))  # K matrix for a single time step.

    kt = mean_actions.reshape((T, 1))
    # print("Mean actions: %s" % kt, flush=True)

    K = np.tile(Kt[None, :, :], (T, 1, 1))  # Controller gains matrix.
    k = kt
    PSig = var_actions.reshape((T, 1, 1))
    cholPSig = np.sqrt(var_actions).reshape((T, 1, 1))
    invPSig = 1.0 / var_actions.reshape((T, 1, 1))

    return LinearGaussianPolicy(K, k, PSig, cholPSig, invPSig)
