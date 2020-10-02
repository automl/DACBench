import numpy as np
from collections import deque
from cma.evolution_strategy import CMAEvolutionStrategy, CMAOptions
from gps.proto.gps_pb2 import CUR_LOC, PAST_OBJ_VAL_DELTAS, CUR_PS, CUR_SIGMA, history_DELTAS, PAST_SIGMA
import threading
import concurrent.futures

from daclib import AbstractEnv

# IDEA: if we ask cma instead of ask_eval, we could make this parallel
def _norm(x): return np.sqrt(np.sum(np.square(x)))
class CMAESEnv(AbstractEnv):
    def __init__(self, config):#dim, init_loc, init_sigma, init_popsize, history_len, fcn=None, hpolib=False, benchmark=None):
        super(CMAESEnv, self).__init__(config)
        #self.hpolib = hpolib
        #self.benchmark = benchmark
        self.b = None
        self.bounds = [None, None]
        self.fbest = None
        self.history_len = config.history_len
        self.historys = deque(maxlen=history_len)
        self.past_obj_vals = deque(maxlen=history_len)
        self.past_sigma = deque(maxlen=history_len)
        self.solutions = None
        self.func_values = []
        self.lock = threading.Lock()

    # action is of shape (dU,)
    def step(self, action):
        done = super(CMAESEnv, self).step_()
        self.historys.append([self.f_difference, self.velocity])
        done = done or self.es.stop()
        if not done:
            """Moves forward in time one step"""
            sigma = action
            self.es.tell(self.solutions, self.func_values)
            self.es.sigma = max(sigma, 0.05)
            self.solutions, self.func_values = self.es.ask_and_eval(self.fcn)

        self.f_difference = np.nan_to_num(np.abs(np.amax(self.func_values) - self.cur_obj_val)/float(self.cur_obj_val))
        self.velocity = np.nan_to_num(np.abs(np.amin(self.func_values) - self.cur_obj_val)/float(self.cur_obj_val))
        self.fbest = min(self.es.best.f, np.amin(self.func_values))

        self.past_obj_vals.append(self.cur_obj_val)
        self.past_sigma.append(self.cur_sigma)
        self.cur_ps = _norm(self.es.adapt_sigma.ps) / self.chi_N - 1
        self.cur_loc = self.es.best.x
        self.cur_sigma = self.es.sigma
        self.cur_obj_val = self.es.best.f
        return self.get_state(), , done, {}

    def reset(self):
        super(CMAESEnv, self).reset_()
        self.history.clear()
        self.past_obj_vals.clear()
        self.past_sigma.clear()
        self.cur_loc = self.instance[0]
        self.cur_sigma = self.instance[1]
        self.init_popsize = self.instance[2]
        self.dim = self.instance[3]
        if len(self.instance) > 4:
            self.fcn = self.instance[4]
        else:
            self.fcn = None

        self.func_values = []
        self.f_vals = deque(maxlen=self.init_popsize)
        self.es = CMAEvolutionStrategy(self.cur_loc, self.init_sigma, {'popsize': self.init_popsize, 'bounds': self.bounds})
        self.solutions, self.func_values = self.es.ask_and_eval(self.fcn)
        self.fbest = self.func_values[np.argmin(self.func_values)]
        self.f_difference = np.abs(np.amax(self.func_values) - self.cur_obj_val)/float(self.cur_obj_val)
        self.velocity = np.abs(np.amin(self.func_values) - self.cur_obj_val)/float(self.cur_obj_val)
        self.es.mean_old = self.es.mean
        self.history.append([self.f_difference, self.velocity])


    def get_state(self):
        past_obj_val_deltas = []
        for i in range(1,len(self.past_obj_vals)):
            past_obj_val_deltas.append((self.past_obj_vals[i] - self.past_obj_vals[i-1]+1e-3) / float(self.past_obj_vals[i-1]))
        if len(self.past_obj_vals) > 0:
            past_obj_val_deltas.append((self.cur_obj_val - self.past_obj_vals[-1]+1e-3)/ float(self.past_obj_vals[-1]))
        past_obj_val_deltas = np.array(past_obj_val_deltas).reshape(-1)

        history_deltas = []
        for i in range(len(self.history)):
            history_deltas.append(self.history[i])
        history_deltas = np.array(history_deltas).reshape(-1)
        past_sigma_deltas = []
        for i in range(len(self.past_sigma)):
            past_sigma_deltas.append(self.past_sigma[i])
        past_sigma_deltas = np.array(past_sigma_deltas).reshape(-1)
        past_obj_val_deltas = np.hstack((np.zeros((self.history_len-past_obj_val_deltas.shape[0],)), past_obj_val_deltas))
        history_deltas = np.hstack((np.zeros((self.history_len*2-history_deltas.shape[0],)), history_deltas))
        past_sigma_deltas = np.hstack((np.zeros((self.history_len-past_sigma_deltas.shape[0],)), past_sigma_deltas))

        cur_loc = self.cur_loc
        cur_ps = self.cur_ps
        cur_sigma = self.cur_sigma

        state = {CUR_LOC: cur_loc,
                 PAST_OBJ_VAL_DELTAS: past_obj_val_deltas,
                 CUR_PS: cur_ps,
                 CUR_SIGMA: cur_sigma,
                 history_DELTAS: history_deltas,
                 PAST_SIGMA: past_sigma_deltas
                }
        return state
