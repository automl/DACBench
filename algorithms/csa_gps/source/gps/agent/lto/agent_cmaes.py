from copy import deepcopy
import numpy as np
from gps.agent.agent import Agent
from gps.proto.gps_pb2 import ACTION
from gps.sample.sample import Sample
from dacbench.benchmarks import CMAESBenchmark


class AgentCMAES(Agent):
    def __init__(self, hyperparams):
        Agent.__init__(self, hyperparams)

        self._setup_conditions()
        self._setup_worlds()

    def _setup_conditions(self):
        self.history_len = self._hyperparams["history_len"]
        self.popsize = self._hyperparams["popsize"]

    def _setup_worlds(self):
        bench = CMAESBenchmark()
        bench.read_instance_set()
        instances = bench.config.instance_set
        bench.config.popsize = self.popsize
        bench.config.hist_length = self.history_len
        bench.config.observation_space_args = [
            {
                "current_loc": spaces.Box(
                    low=-np.inf, high=np.inf, shape=np.arange(INPUT_DIM).shape
                ),
                "past_deltas": spaces.Box(
                    low=-np.inf, high=np.inf, shape=np.arange(HISTORY_LENGTH).shape
                ),
                "current_ps": spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
                "current_sigma": spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
                "history_deltas": spaces.Box(
                    low=-np.inf, high=np.inf, shape=np.arange(HISTORY_LENGTH * 2).shape
                ),
                "past_sigma_deltas": spaces.Box(
                    low=-np.inf, high=np.inf, shape=np.arange(HISTORY_LENGTH).shape
                ),
            }
        ]
        self._worlds = []
        for i in instances:
            bench.config.instance_set = [i]
            self._worlds.append(bench.get_environment())
        self.x0 = []

        for i in range(self.conds):
            self._worlds[i].reset()  # Get noiseless initial state
            x0 = self.get_vectorized_state(self._worlds[i].get_state())
            self.x0.append(x0)

    def sample(
        self,
        policy,
        condition,
        start_policy=None,
        verbose=False,
        save=True,
        noisy=True,
        ltorun=False,
        guided_steps=0,
        t_length=None,
    ):
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
        new_sample = self._init_sample(self.get_vectorized_state(state))
        new_sample.trajectory.append(self._worlds[condition].fbest)
        U = np.zeros([t_length, self.dU])
        if noisy:
            noise = np.random.randn(t_length, self.dU)
        else:
            noise = np.zeros((t_length, self.dU))
        policy.reset()  # To support non-Markovian policies
        for t in range(t_length):
            es = self._worlds[condition].es
            f_vals = self._worlds[condition].func_values
            obs_t = new_sample.get_obs(t=t)
            X_t = self.get_vectorized_state(state, condition)
            if np.any(np.isnan(X_t)):
                print("X_t: %s" % X_t)
            if ltorun and t < guided_steps * t_length and start_policy != None:
                U[t, :] = start_policy.act(es, f_vals, obs_t, t, noise[t, :])
            else:
                U[t, :] = policy.act(X_t, obs_t, t, noise[t, :], es, f_vals)
            if (t + 1) < t_length:
                next_action = U[t, :]  # * es.sigma
                state, reward, done, _ = self._worlds[condition].step(next_action)
                self._set_sample(new_sample, state, t)
            new_sample.trajectory.append(reward)
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
            sample.set(sensor, np.array(X[sensor]), t=t + 1)
