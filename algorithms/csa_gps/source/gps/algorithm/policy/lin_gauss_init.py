""" Initializations for linear Gaussian controllers. """
import copy
import numpy as np
import scipy as sp
from gps.algorithm.policy.config import INIT_LG
from gps.algorithm.policy.csa_policy import CSAPolicy
from gps.algorithm.policy.lin_gauss_policy import LinearGaussianPolicy
from gps.agent.lto.cmaes_world import CMAESWorld

def init_cmaes_controller(hyperparams, agent):

    config = copy.deepcopy(INIT_LG)
    config.update(hyperparams)

    dX, dU = config['dX'], config['dU']
    T = config['T']
    cur_cond_idx = config['cur_cond_idx']
    history_len = agent.history_len
    fcn = agent.fcns[cur_cond_idx]
    popsize = agent.popsize
    if 'fcn_obj' in fcn:
        fcn_obj = fcn['fcn_obj']
    else:
        fcn_obj = None
    hpolib = False
    if 'hpolib' in fcn:
        hpolib = True
    benchmark = None
    if 'benchmark' in fcn:
        benchmark = fcn['benchmark']
    # Create new world to avoiding changing the state of the original world
    world = CMAESWorld(dim=fcn['dim'], init_loc=fcn['init_loc'], init_sigma=fcn['init_sigma'], init_popsize=popsize, history_len=history_len, fcn=fcn_obj, hpolib=hpolib, benchmark=benchmark)

    if config['verbose']:
        print("Finding Initial Linear Gaussian Controller")
    action_mean = []
    action_var = []
    for i in range(25):
            f_values=[]
            cur_policy = CSAPolicy(T=T)

            world.reset_world()
            world.run()
            for t in range(T):
                X_t = agent.get_vectorized_state(world.get_state(), cur_cond_idx)
                es = world.es
                f_vals = world.func_values
                U_t = cur_policy.act(X_t, None, t, np.zeros((dU,)), es, f_vals)
                world.run_next(U_t)
                f_values.append(U_t)
            action_mean.append(f_values)# np.mean(f_values, axis=0))
            action_var.append(f_values)# np.mean(f_values, axis=0))
    mean_actions = np.mean(action_mean, axis=0)
    var_actions = np.std(action_var, axis=0)
    np.place(var_actions, var_actions==0, config["init_var"]) 
    Kt = np.zeros((dU, dX)) # K matrix for a single time step.

    kt = mean_actions.reshape((T,1))
    #print("Mean actions: %s" % kt, flush=True)

    K = np.tile(Kt[None,:,:], (T, 1, 1))     # Controller gains matrix.
    k = kt
    PSig = var_actions.reshape((T, 1, 1))
    cholPSig = np.sqrt(var_actions).reshape((T, 1, 1))
    invPSig = 1./var_actions.reshape((T, 1, 1))

    return LinearGaussianPolicy(K, k, PSig, cholPSig, invPSig)



