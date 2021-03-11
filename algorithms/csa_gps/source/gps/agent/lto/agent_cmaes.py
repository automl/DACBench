from copy import deepcopy
import numpy as np
from gps.agent.agent import Agent
from gps.proto.gps_pb2 import ACTION
from gps.sample.sample import Sample
from dacbench.benchmarks import CMAESBenchmark

def rename_state_keys(state):
    state = {1: state["current_loc"][0],
            2: state["past_deltas"],
            3: state["current_ps"][0],
            4: state["current_sigma"][0],
            5: state["history_deltas"],
            6: state["past_sigma_deltas"]
            }
    return state

class AgentCMAES(Agent):
    
    def __init__(self, hyperparams):
        Agent.__init__(self, hyperparams)
        
        self._setup_conditions()
        self._setup_worlds()
    
    def _setup_conditions(self):
        self.conds = self._hyperparams['conditions']
        self.fcns = self._hyperparams['fcns']
        self.history_len = self._hyperparams['history_len']
        self.init_sigma = self._hyperparams['init_sigma']
        self.popsize = self._hyperparams['popsize']
        
    def _setup_worlds(self):
        fcn = []
        hpolib = False
        for i in range(self.conds):
            if 'fcn_obj' in self.fcns[i]:
                fcn.append(self.fcns[i]['fcn_obj'])
            else:
                fcn.append(None)
            if 'hpolib' in self.fcns[i]:
                hpolib = True
        benchmark = None
        bench = CMAESBenchmark()
        env = bench.get_environment()
        inst_set = env.instance_set
        self._worlds = []
        for i in inst_set.values():
            bench.instance_set = {0: i}
            self._worlds.append(bench.get_environment())
        #if 'benchmark' in self.fcns[0]:
        #    benchmark = self.fcns[0]['benchmark']
        #self._worlds = [CMAESWorld(self.fcns[i]['dim'], self.fcns[i]['init_loc'], self.fcns[i]['init_sigma'], self.popsize, self.history_len, fcn=fcn[i], hpolib=hpolib, benchmark=benchmark) for i in range(self.conds)]
        self.x0 = []
        for i in range(self.conds):
            state = self._worlds[i].reset()
            state = rename_state_keys(state)
            #self._worlds[i].run()      # Get noiseless initial state
            x0 = self.get_vectorized_state(state)
            self.x0.append(x0)

    def sample(self, policy, condition, start_policy=None, verbose=False, save=True, noisy=True, ltorun=False, guided_steps=0, t_length=None):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.

        Args:
            policy: Policy to to used in the trial.
            condition (int): Which condition setup to run.
            verbose (boolean): Whether or not to plot the trial (not used here).
            save (boolean): Whether or not to store the trial into the samples.
            noisy (boolean): Whether or not to use noise during sampling.
        """
        if t_length == None:
            t_length = self.T
        state = self._worlds[condition].reset()
        state = rename_state_keys(state)
        new_sample = self._init_sample(state)
        #self._set_sample(new_sample, self._worlds[condition].get_state(), t=0)
        new_sample.trajectory.append(self._worlds[condition].fbest)
        U = np.zeros([t_length, self.dU])
        if noisy:
            noise = np.random.randn(t_length, self.dU)
        else:
            noise = np.zeros((t_length, self.dU))
        policy.reset()      # To support non-Markovian policies
        for t in range(t_length):
            es = self._worlds[condition].es
            f_vals = self._worlds[condition].func_values
            obs_t = new_sample.get_obs(t=t)
            X_t = self.get_vectorized_state(rename_state_keys(self._worlds[condition].get_state(None)), condition)
            if np.any(np.isnan(X_t)):
                print("X_t: %s" % X_t)
            if ltorun and t < guided_steps * t_length and start_policy != None:
                U[t,:] = start_policy.act(es, f_vals, obs_t, t, noise[t,:])
            else:
                U[t, :] = policy.act(X_t, obs_t, t, noise[t, :],es, f_vals)
            if (t+1) < t_length:
                next_action = U[t, :] #* es.sigma
                state, reward, done, _ = self._worlds[condition].step(next_action)
                self._set_sample(new_sample, state, t)
                print(-reward)
            new_sample.trajectory.append(-reward)
        new_sample.set(ACTION, U)
        policy.finalize()
        if save:
            self._samples[condition].append(new_sample)
        return new_sample
    
    def _init_sample(self, init_X):
        """
        Construct a new sample and fill in the first time step.
        """
        sample = Sample(self)
        self._set_sample(sample, init_X, -1)
        return sample

    def _set_sample(self, sample, X, t):
        for sensor in X.keys():
            sample.set(sensor, np.array(X[sensor]), t=t+1)
