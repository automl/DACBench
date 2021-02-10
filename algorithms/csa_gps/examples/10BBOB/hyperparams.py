import tensorflow as tf

import os.path
from datetime import datetime
import numpy as np
import cma
from cma import bbobbenchmarks as bn
import gps
from gps import __file__ as gps_filepath
from gps.agent.lto.agent_cmaes import AgentCMAES
from gps.agent.lto.cmaes_world import CMAESWorld
from gps.algorithm.algorithm import Algorithm
from gps.algorithm.cost.cost import Cost
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.traj_opt.traj_opt import TrajOpt
from gps.algorithm.policy_opt.policy_opt import PolicyOpt
from gps.algorithm.policy_opt.lto_model import fully_connected_tf_network
from gps.algorithm.policy.lin_gauss_init import init_cmaes_controller
from gps.proto.gps_pb2 import CUR_LOC, PAST_OBJ_VAL_DELTAS, CUR_SIGMA, CUR_PS, PAST_LOC_DELTAS,PAST_SIGMA, ACTION
from gps.algorithm.cost.cost_utils import RAMP_CONSTANT

try:
   import cPickle as pickle
except:
   import pickle
import copy



session = tf.Session()
history_len = 40

num_fcns = 100
train_fcns = range(num_fcns)
test_fcns = range(num_fcns-20, num_fcns)

fcn_ids = [12, 11, 2, 23, 15, 8, 17, 20, 1, 16]
fcn_names = ["BentCigar", "Discus", "Ellipsoid", "Katsuura", "Rastrigin", "Rosenbrock", "Schaffers", "Schwefel", "Sphere", "Weierstrass"]

input_dim = 10
num_inits_per_fcn = 1
cur_dir = os.path.dirname(os.path.abspath(__file__))
 
init_sigma_test = [1.28, 0.38, 1.54, 1.18, 0.1, 1.66, 0.33, 0.1, 1.63, 0.1]

#initialize the initial locations of the optimization trajectories
init_locs = list(np.random.randn(num_fcns-10, input_dim)) 
#append the initial locations of the conditions in the test set
for i in range(20):
    init_locs.append([0]*10)

#initialize the initial sigma(step size) values
init_sigmas = list(np.random.rand(num_fcns-10)) + [0.5]*10

fcn_objs = []
fcns = []
for i in range(num_fcns//len(fcn_ids)):
    #for k in range(7,9):
    for i in fcn_ids:
        fcn_objs.append(bn.instantiate(i)[0])

for i,function in enumerate(fcn_objs):
    fcns.append({'fcn_obj': function, 'dim': input_dim, 'init_loc': list(init_locs[i]), 'init_sigma': init_sigmas[i]})

SENSOR_DIMS = {
    PAST_OBJ_VAL_DELTAS: history_len,
    CUR_PS: 1,
    CUR_SIGMA : 1,
    ACTION: 1,
    PAST_SIGMA: history_len
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../examples/10BBOB' + '/'


common = {
    'experiment_name': 'CMA_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'plot_filename': EXP_DIR + 'plot',
    'log_filename': EXP_DIR + 'log_data.txt',
    'conditions': num_fcns,
    'train_conditions': train_fcns,
    'test_conditions': test_fcns,
    'test_functions': test_functions
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentCMAES,
    'world' : CMAESWorld,
    'init_sigma': 0.3,
    'popsize': 10,
    'n_min':10,
    'max_nfe': 200000,
    'substeps': 1,
    'conditions': common['conditions'],
    'dt': 0.05,
    'T': 50,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [PAST_OBJ_VAL_DELTAS, CUR_SIGMA, CUR_PS, PAST_SIGMA],
    'obs_include': [PAST_OBJ_VAL_DELTAS, CUR_PS, PAST_SIGMA, CUR_SIGMA],
    'history_len': history_len,
    'fcns': fcns
}

algorithm = {
    'type': Algorithm,
    'conditions': common['conditions'],
    'train_conditions': train_fcns,
    'test_conditions': test_fcns,
    'test_functions': fcn_names,
    'iterations': 15,
    'inner_iterations': 4,
    'policy_dual_rate': 0.2,
    'init_pol_wt': 0.01,
    'ent_reg_schedule': 0.0,
    'fixed_lg_step': 3,
    'kl_step': 0.2,
    'min_step_mult': 0.01,
    'max_step_mult': 10.0,
    'sample_decrease_var': 0.05,
    'sample_increase_var': 0.1,
    'policy_sample_mode': 'replace',
    'exp_step_lower': 2,
    'exp_step_upper': 2
}

algorithm['init_traj_distr'] = {
    'type': init_cmaes_controller,
    'init_var': 0.01,
    'dt': agent['dt'],
    'T': agent['T']
}

algorithm['cost'] = {
    'type': Cost,
    'ramp_option': RAMP_CONSTANT,
    'wp_final_multiplier': 1.0,
    'weight': 1.0,
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-3,     # Increase this if Qtt is not PD during DGD
    'clipping_thresh': None,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 20,
        'max_samples': 20,
        'strength': 1.0         # How much weight to give to prior relative to samples
    }
}

algorithm['traj_opt'] = {
    'type': TrajOpt,
}

algorithm['policy_opt'] = {
    'type': PolicyOpt,
    'network_model': fully_connected_tf_network,
    'iterations': 20000,
    'init_var': 0.01,
    'batch_size': 25,
    'solver_type': 'adam',
    'lr': 0.0001,
    'lr_policy': 'fixed',
    'momentum': 0.9,
    'weight_decay': 0.005,
    'use_gpu': 0,
    'weights_file_prefix': EXP_DIR + 'policy',
    'network_params': {
        'obs_include': agent['obs_include'],
        'sensor_dims': agent['sensor_dims'],
        'dim_hidden': [50, 50]
    }
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 20,
    'max_samples': 20,
    'strength': 1.0,
    'clipping_thresh': None,
    'init_regularization': 1e-3,
    'subsequent_regularization': 1e-3
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 25,
    'common': common,
    'agent': agent,
    'algorithm': algorithm,
    'train_conditions': train_fcns,
    'test_conditions': test_fcns,
    'test_functions': fcn_names,
    'policy_path':  EXP_DIR + 'data_files/policy_itr_14.pkl'
}

