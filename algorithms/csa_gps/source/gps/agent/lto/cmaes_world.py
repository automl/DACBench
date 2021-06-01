import numpy as np
from collections import deque
from cma.evolution_strategy import CMAEvolutionStrategy, CMAOptions
from gps.proto.gps_pb2 import (
    CUR_LOC,
    PAST_OBJ_VAL_DELTAS,
    CUR_PS,
    CUR_SIGMA,
    PAST_LOC_DELTAS,
    PAST_SIGMA,
)
import threading
import concurrent.futures


def _norm(x):
    return np.sqrt(np.sum(np.square(x)))


class CMAESWorld(object):
    def __init__(
        self,
        dim,
        init_loc,
        init_sigma,
        init_popsize,
        history_len,
        fcn=None,
        hpolib=False,
        benchmark=None,
    ):
        if fcn is not None:
            self.fcn = fcn
        else:
            self.fcn = None
        self.hpolib = hpolib
        self.benchmark = benchmark
        self.b = None
        self.bounds = [None, None]
        self.dim = dim
        self.init_loc = init_loc
        self.init_sigma = init_sigma
        self.init_popsize = init_popsize
        self.fbest = None
        self.history_len = history_len
        self.past_locs = deque(maxlen=history_len)
        self.past_obj_vals = deque(maxlen=history_len)
        self.past_sigma = deque(maxlen=history_len)
        self.solutions = None
        self.func_values = []
        self.f_vals = deque(maxlen=self.init_popsize)
        self.lock = threading.Lock()
        self.chi_N = dim ** 0.5 * (1 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))

    def run(self, batch_size="all", ltorun=False):
        """Initiates the first time step"""
        # self.fcn.new_sample(batch_size=batch_size)
        self.cur_loc = self.init_loc
        self.cur_sigma = self.init_sigma
        self.cur_ps = 0
        self.es = CMAEvolutionStrategy(
            self.cur_loc,
            self.init_sigma,
            {"popsize": self.init_popsize, "bounds": self.bounds},
        )
        self.solutions, self.func_values = self.es.ask_and_eval(self.fcn)
        self.fbest = self.func_values[np.argmin(self.func_values)]
        self.f_difference = np.abs(
            np.amax(self.func_values) - self.cur_obj_val
        ) / float(self.cur_obj_val)
        self.velocity = np.abs(np.amin(self.func_values) - self.cur_obj_val) / float(
            self.cur_obj_val
        )
        self.es.mean_old = self.es.mean
        self.past_locs.append([self.f_difference, self.velocity])

    # action is of shape (dU,)
    def run_next(self, action):
        self.past_locs.append([self.f_difference, self.velocity])
        if not self.es.stop():
            """Moves forward in time one step"""
            sigma = action
            self.es.tell(self.solutions, self.func_values)
            self.es.sigma = min(max(sigma, 0.05), 10)
            self.solutions, self.func_values = self.es.ask_and_eval(self.fcn)

        self.f_difference = np.nan_to_num(
            np.abs(np.amax(self.func_values) - self.cur_obj_val)
            / float(self.cur_obj_val)
        )
        self.velocity = np.nan_to_num(
            np.abs(np.amin(self.func_values) - self.cur_obj_val)
            / float(self.cur_obj_val)
        )
        self.fbest = min(self.es.best.f, np.amin(self.func_values))

        self.past_obj_vals.append(self.cur_obj_val)
        self.past_sigma.append(self.cur_sigma)
        self.cur_ps = _norm(self.es.adapt_sigma.ps) / self.chi_N - 1
        self.cur_loc = self.es.best.x
        self.cur_sigma = self.es.sigma
        self.cur_obj_val = self.es.best.f

    def reset_world(self):
        self.past_locs.clear()
        self.past_obj_vals.clear()
        self.past_sigma.clear()
        self.cur_loc = self.init_loc
        self.cur_sigma = self.init_sigma
        self.cur_ps = 0
        self.func_values = []

    def get_state(self):
        past_obj_val_deltas = []
        for i in range(1, len(self.past_obj_vals)):
            past_obj_val_deltas.append(
                (self.past_obj_vals[i] - self.past_obj_vals[i - 1] + 1e-3)
                / float(self.past_obj_vals[i - 1])
            )
        if len(self.past_obj_vals) > 0:
            past_obj_val_deltas.append(
                (self.cur_obj_val - self.past_obj_vals[-1] + 1e-3)
                / float(self.past_obj_vals[-1])
            )
        past_obj_val_deltas = np.array(past_obj_val_deltas).reshape(-1)

        past_loc_deltas = []
        for i in range(len(self.past_locs)):
            past_loc_deltas.append(self.past_locs[i])
        past_loc_deltas = np.array(past_loc_deltas).reshape(-1)
        past_sigma_deltas = []
        for i in range(len(self.past_sigma)):
            past_sigma_deltas.append(self.past_sigma[i])
        past_sigma_deltas = np.array(past_sigma_deltas).reshape(-1)
        past_obj_val_deltas = np.hstack(
            (
                np.zeros((self.history_len - past_obj_val_deltas.shape[0],)),
                past_obj_val_deltas,
            )
        )
        past_loc_deltas = np.hstack(
            (
                np.zeros((self.history_len * 2 - past_loc_deltas.shape[0],)),
                past_loc_deltas,
            )
        )
        past_sigma_deltas = np.hstack(
            (
                np.zeros((self.history_len - past_sigma_deltas.shape[0],)),
                past_sigma_deltas,
            )
        )

        cur_loc = self.cur_loc
        cur_ps = self.cur_ps
        cur_sigma = self.cur_sigma

        state = {
            CUR_LOC: cur_loc,
            PAST_OBJ_VAL_DELTAS: past_obj_val_deltas,
            CUR_PS: cur_ps,
            CUR_SIGMA: cur_sigma,
            PAST_LOC_DELTAS: past_loc_deltas,
            PAST_SIGMA: past_sigma_deltas,
        }
        return state
