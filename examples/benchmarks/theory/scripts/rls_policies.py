import numpy as np
from dacbench.envs import RLSEnv, RLSEnvDiscreteK
import os
import sys
sys.path.append(os.path.dirname(__file__))
from ddqn_local.ddqn import DQN
import pprint
   
class RandomPolicy:
    def __init__(self, env, name="random"):
        self.name = name
        self.env = env        
    def get_next_action(self, obs):
        action = self.env.action_space.sample()
        return action
    
class RLSOptimalPolicy:
    def __init__(self, env, name="RLSOptimal"):
        self.name = name
        self.env = env        
    def get_next_action(self, obs):
        k = int(self.env.n / (self.env.x.fitness+1))
        return k
    
class RLSFixedOnePolicy:
    def __init__(self, env, name="RLSFixedOne"):
        self.name = name
        self.env = env        
    def get_next_action(self, obs):        
        return 1
    
class DQNPolicy:
    def __init__(self, env, model, name="RLSDQNPolicy"):
        """
        model: trained model
        """                
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.agent = DQN(state_dim, action_dim, 1, env=env, eval_env=env, out_dir="./")
        self.agent.load(model)
    def get_next_action(self, obs):
        return self.agent.get_action(obs, 0)
    
class RLSOptimalDiscretePolicy:
    def __init__(self, env, name="RLSOptimalDiscrete"):        
        self.name = name
        self.env = env
        assert "action_choices" in env.unwrapped.__dict__        
        env.reset()
        self.acts = env.action_choices
        # string of all actions
        self.acts_str = str(self.acts)
        script_dir = os.path.dirname(__file__)
        self.positions_dict = self.parse_rls_optimal_policies(fn=script_dir + '/optimal_policies.txt')        
        sn = str(env.n)
        assert sn in self.positions_dict
        assert self.acts_str in self.positions_dict[sn]
        self.positions = self.positions_dict[sn][self.acts_str]
        
    def get_next_action(self, obs):                    
        fx = self.env.x.fitness
        act = None        
        for i in range(len(self.positions)-1, -1, -1):            
            if self.positions[i] >= fx:
                act = i      
                break
        #print(f"{fx:3d}: {self.acts[act]:3d}")
        assert act is not None, f"ERROR: {fx}"
        return act    
    
    def parse_rls_optimal_policies(self, fn='./optimal_policies.txt'):        
        with open(fn, "rt") as f:
            ls = [s.replace("\n","") for s in f.readlines()]
            ls = [s for s in ls if s!=""]
        positions_dict = {}
        for s in ls:
            ls1 = s.split(';')
            n = int(ls1[0])
            ks = [x.strip() for x in ls1[1].split(',') if x.strip()!=""]
            pos = [x.strip() for x in ls1[2].split(',') if x.strip()!=""]
            assert len(ks)==len(pos), f"ERROR with {s} ({len(ks)} {len(pos)})"
            ks = '[' + ', '.join([str(x) for x in ks]) + ']'
            pos = [int(x) for x in pos]
            sn = str(self.env.n)
            if sn not in positions_dict:
                positions_dict[sn] = {}
            positions_dict[sn][ks] = pos            
                
        #pprint.pprint(positions_dict)
        return positions_dict
             
        
