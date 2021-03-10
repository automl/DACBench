from os import fdatasync
import numpy as np
from copy import deepcopy
import logging
from collections import deque

import chainerrl
import chainer

import sys
import os

from dacbench import AbstractEnv

class BinaryProblem:
    """
    An abstract class for an individual in binary representation
    """
    def __init__(self, n, val=None, rng=np.random.default_rng()):
        if val is not None:
            assert isinstance(val, bool)
            self.data = np.array([val] * n)
        else:
            self.data = rng.choice([True,False], size=n) 
        self.n = n
        self.fitness = self.eval()
        

    def is_optimal(self):
        pass


    def get_optimal(self):
        pass


    def eval(self):
        pass        


    def get_fitness_change_after_mutation(self, locs):
        """
        Calculate the change in fitness after flipping the bits at positions locs

        Parameters
        -----------
            locs: 1d-array
                positions where bits are flipped

        Returns: int
        -----------
            objective after mutation - objective before mutation
        """
        raise NotImplementedError


    def get_fitness_after_crossover(self, xprime, locs_x, locs_xprime):
        """
        Calculate fitness of the child when crossover with xprime, without doing the actual crossover

        Parameters
        -----------
            xprime: 1d boolean array
                the individual to crossover with
            locs_x: 1d boolean/integer array
                positions where we keep current bits of self
            locs_xprime: : 1d boolean/integer array
                positions where we change to xprime's bits

        Returns: int
        -----------
            fitness of the new individual after crossover

        """
        raise NotImplementedError


    def mutate(self, p, n_offsprings, rng=np.random.default_rng()):
        """
        Draw l ~ binomial(n, p), l>0
        Generate n_offsprings children by flipping exactly l bits and select the best one

        Parameters
        -----------
            p: float
                mutation probability, in range of [0,1]
            n_offsprings: int
                number of mutated children

        Returns
        ----------- 
            the best child (maximum fitness), its fitness and number of evaluations used        
        """

        assert p>=0

        if p==0:
            return self, self.fitness, 0

        l = 0
        while l==0:
            l = rng.binomial(self.n, p)                
        
        best_delta = -self.n
        best_locs = None
        for i in range(n_offsprings):
            locs = rng.choice(self.n, size=l, replace=False)        
            delta = self.get_fitness_change_after_mutation(locs)
            if delta > best_delta:
                best_locs = locs
                best_delta = delta                       

        best_child = deepcopy(self)
        best_child.data[best_locs] = ~best_child.data[best_locs]
        best_child.fitness = self.fitness + best_delta

        return best_child, best_child.fitness, n_offsprings


    def crossover(self, xprime, p, n_offsprings, 
                    include_xprime=True, count_different_inds_only=True,
                    rng=np.random.default_rng()):
        """
        Generate n_offsprings children using crossover on self and xprime, and return the best one
            Crossover operator: for each bit, taking value from xprime with probability p and from self with probability 1-p

        Parameters
        ---------
            xprime: 1d boolean numpy array
                the individual to crossover with
            p: float
                crossover bias, in range of [0,1]
            include_xprime: boolean
                if True, include xprime in the selection for the best individual after all crossovers
            count_different_inds_only: boolean
                if True, only count an evaluation of a child if it is different from both of its parents (self and xprime)
            rng: numpy random generator   

        Returns
        ---------  
            the best child (maximum fitness), its fitness and number of evaluations used 

        """
        assert p <= 1
        
        if p == 0:
            if include_xprime:
                return xprime, xprime.fitness, 0
            else:
                return self, self.fitness, 0            

        if include_xprime:
            best_val = xprime.fitness
        else:
            best_val = -1            
        best_locs = None

        n_evals = 0
        ls = rng.binomial(self.n, p, size=n_offsprings)
        locs_x = np.empty(self.n, dtype=np.bool)        
        for l in ls:                   
            locs_xprime = rng.choice(self.n, l, replace=False)
            locs_x.fill(True)            
            locs_x[locs_xprime] = False
            val = self.get_fitness_after_crossover(xprime, locs_x, locs_xprime)            

            if (val != self.fitness) and (val!=xprime.fitness):
                n_evals += 1
            elif (not np.array_equal(xprime.data[locs_xprime], self.data[locs_xprime])) and (not np.array_equal(self.data[locs_x], xprime.data[locs_x])):            
                n_evals += 1            

            if val > best_val:
                best_val = val
                best_locs = locs_xprime
                        
        if best_locs is not None:
            child = deepcopy(self)
            child.data[best_locs] = xprime.data[best_locs]
            child.fitness = best_val
        else:
            child = xprime

        if not count_different_inds_only:
            n_evals = n_offsprings

        return child, best_val, n_evals


class OneMax(BinaryProblem):
    """
    An individual for OneMax problem
    The aim is to maximise the number of 1 bits
    """

    def eval(self):
        self.fitness = self.data.sum()
        return self.fitness

    def is_optimal(self):
        return self.data.all()

    def get_optimal(self):
        return self.n

    def get_fitness_change_after_mutation(self, locs):        
        # f(x_new) = f(x) + l - 2 * sum_of_flipped_block
        return len(locs) - 2 * self.data[locs].sum()

    def get_fitness_after_crossover(self, xprime, locs_x, locs_xprime):        
        return self.data[locs_x].sum() + xprime.data[locs_xprime].sum()
        

class LeadingOne(BinaryProblem):    
    """
    An individual for LeadingOne problem
    The aim is to maximise the number of leading (and consecutive) 1 bits in the string
    """

    def eval(self):
        k = self.data.argmin()
        if self.data[k]:
            self.fitness = self.n
        else:
            self.fitness = k
        return self.fitness

    def is_optimal(self):
        return self.data.all()  

    def get_optimal(self):
        return self.n

    def get_fitness_change_after_mutation(self, locs):        
        min_loc = locs.min()
        if min_loc < self.fitness:
            return min_loc - self.fitness
        elif min_loc > self.fitness:
            return 0
        else:
            old_fitness = self.fitness
            self.data[locs] = ~self.data[locs]
            new_fitness = self.eval()
            delta_fitness = new_fitness - old_fitness
            self.data[locs] = ~self.data[locs]
            self.fitness = old_fitness
            return delta_fitness

    def get_fitness_after_crossover(self, xprime, locs_x, locs_xprime):        
        """
        this implementation should be improved
        """
        child = deepcopy(self)
        child.data[locs_xprime] = xprime.data[locs_xprime]
        child.eval()
        return child.fitness

HISTORY_LENGTH = 5

class OneLLEnv(AbstractEnv):
    """
    Environment for (1+(lbd, lbd))-GA
    for both OneMax and LeadingOne problems
    """

    def __init__(self, config) -> None:
        """
        Initialize OneLLEnv

        Parameters
        -------
        config : objdict
            Environment configuration
        """
        super(OneLLEnv, self).__init__(config)        
        self.logger = logging.getLogger(self.__str__())     

        self.name = config.name   
        
        # parameters of OneLL-GA
        self.problem = globals()[config.problem]
        self.include_xprime = config.include_xprime
        self.count_different_inds_only = config.count_different_inds_only
      
        # names of all variables in a state
        self.state_description = config.observation_description
        self.state_var_names = [s.strip() for s in config.observation_description.split(',')]

        # functions to get values of the current state from histories 
        # (see reset() function for those history variables)        
        self.state_functions = []
        for var_name in self.state_var_names:
            if var_name == 'n':
                self.state_functions.append(lambda: self.n)
            elif var_name in ['lbd','lbd1','lbd2', 'p', 'c']:
                self.state_functions.append(lambda: vars(self)['history_' + var_name][-1])
            elif "_{t-" in var_name:
                k = int(var_name.split("_{t-")[1][:-1]) # get the number in _{t-<number>}
                name = var_name.split("_{t-")[0] # get the variable name (lbd, lbd1, etc)
                self.state_functions.append(lambda: vars(self)['history_' + name][-k])
            elif var_name == "f(x)":
                self.state_functions.append(lambda: self.history_fx[-1])
            elif var_name == "delta f(x)":
                self.state_functions.append(lambda: self.history_fx[-1] - self.history_fx[-2])
            else:
                raise Exception("Error: invalid state variable name: " + var_name)
        
        # names of all variables in an action
        self.action_description = config.action_description
        self.action_var_names = [s.strip() for s in config.action_description.split(',')] # names of 
        for name in self.action_var_names:
            assert name in ['lbd', 'lbd1', 'lbd2', 'p', 'c'], "Error: invalid action variable name: " + name

        # the random generator used by OneLL-GA
        if 'seed' in config:
            seed = config.seed
        else:
            seed = None
        self.rng = np.random.default_rng(seed)   

        # for logging
        self.n_eps = 0 # number of episodes done so far        
             


    def reset(self):
        """
        Resets env

        Returns
        -------
        numpy.array
            Environment state
        """        
        super(OneLLEnv, self).reset_()        

        # current problem size (n) & evaluation limit (max_evals)
        self.n = self.instance.size
        self.max_evals = self.instance.max_evals
        self.logger.info("n:%d, max_evals:%d" % (self.n, self.max_evals))

        # create an initial solution
        self.x = self.problem(n=self.instance.size, rng=self.rng)

        # total number of evaluations so far
        self.total_evals = 1        

        # reset histories (not all of those are used at the moment)        
        self.history_lbd = deque([-1]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH) # either this one or the next two (history_lbd1, history_lbd2) are used, depending on our configuration
        self.history_lbd1 = deque([-1]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
        self.history_lbd2 = deque([-1]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
        self.history_p = deque([-1]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
        self.history_c = deque([-1]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
        self.history_fx = deque([self.x.fitness]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)        
        
        return self.get_state()


    def get_state(self):
        return np.asarray([f() for f in self.state_functions])


    def get_onell_params(self, action):
        """
        Get OneLL-GA parameters (lbd1, lbd2, p and c) from an action

        Returns: lbd1, lbd2, p, c
            lbd1: float (will be converted to int in step())
                number of mutated off-springs: in range [1,n]
            lbd2: float (will be converted to int in step())
                number of crossovered off-springs: in range [1,n]
            p: float
                mutation probability
            c: float
                crossover bias
        """
        i = 0
        rs = {}
        if not isinstance(action, np.ndarray):
            action = [action]
        for var_name in self.action_var_names:
            if var_name == 'lbd':
                rs['lbd1'] = rs['lbd2'] = np.clip(action[i], 1, self.n)
            elif 'lbd' in var_name: # lbd1 or lbd2 
                rs[var_name] = np.clip(action[i], 1, self.n)
            else: # must be p or c
                rs[var_name] = np.clip(action[i], 0, 1)
            i+=1

        # if p and c are not set, use the default formula
        if not 'p' in rs.keys():
            rs['p'] = rs['lbd1'] / self.n
        if not 'c' in rs.keys():
            rs['c'] = 1 / rs['lbd1']

        return rs['lbd1'], rs['lbd2'], rs['p'], rs['c']

    
    def step(self, action):
        """
        Execute environment step

        Parameters
        ----------
        action : Box
            action to execute

        Returns
        -------            
            state, reward, done, info
            np.array, float, bool, dict
        """
        super(OneLLEnv, self).step_()                
                
        fitness_before_update = self.x.fitness

        lbd1, lbd2, p, c = self.get_onell_params(action)

        # mutation phase
        xprime, f_xprime, ne1 = self.x.mutate(p, int(lbd1), self.rng)

        # crossover phase
        y, f_y, ne2 = self.x.crossover(xprime, c, int(lbd2), self.include_xprime, self.count_different_inds_only, self.rng)        
        
        # update x
        if self.x.fitness <= y.fitness:
            self.x = y
        
        # update total number of evaluations
        n_evals = ne1 + ne2
        self.total_evals += n_evals

        # check stopping criteria        
        done = (self.total_evals>=self.instance.max_evals) or (self.x.is_optimal())        
        
        # calculate reward        
        reward = self.x.fitness - fitness_before_update - n_evals

        # update histories
        self.history_fx.append(self.x.fitness)
        self.history_lbd1.append(lbd1)
        self.history_lbd2.append(lbd2)
        self.history_lbd.append(lbd1)
        self.history_p.append(p)
        self.history_c.append(c)
        
        if done:
            self.n_eps += 1
            self.logger.info("Episode done: ep:%d, n:%d, obj:%d, evals:%d" % (self.n_eps, self.n, self.x.fitness, self.total_evals))            
        
        return self.get_state(), reward, done, {}


    def plot_agent_prediction(self, agent, dirname):
        """
        Plot agent progress for this particular environment
        """

        # plot 1: f(x) as x-axis, optimal and predicted lbd values as y-axis
        agent.load(dirname)
        obss = np.asarray([[self.n,i] for i in range(self.n)], dtype=np.float32)
        b_state = chainerrl.misc.batch_states(obss, agent.xp, agent.phi)

        if agent.obs_normalizer:
            b_state = agent.obs_normalizer(b_state, update=False)

        with chainer.using_config('train', False), chainer.no_backprop_mode():        
            action_distrib, values = agent.model(b_state)                                

        print(action_distrib['mean'].array)
            

    def close(self) -> bool:
        """
        Close Env

        No additional cleanup necessary

        Returns
        -------
        bool
            Closing confirmation
        """        
        return True



