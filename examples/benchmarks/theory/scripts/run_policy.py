import os
import sys
sys.path.append(os.path.dirname(__file__))
import argparse
import yaml
import pickle

from rls_policies import RandomPolicy, RLSOptimalPolicy, RLSFixedOnePolicy, DQNPolicy, RLSOptimalDiscretePolicy
from utils import make_env, read_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", "-n", type=int, default=10, help='number of episodes')
    parser.add_argument("--bench", "-b",type=str, required=True, help='yml file of benchmark configuration')
    parser.add_argument("--policy", "-p", type=str, required=True, help='policy name')
    parser.add_argument("--policy-params", "-pr", type=str, default=None, help='policy params')
    parser.add_argument("--out", "-o", type=str)
    args = parser.parse_args()
                        
    # create enviroment
    with open(args.bench,'r') as f:
        bench_params = yaml.safe_load(f)
    
    env = None
    # load policy
    if args.policy in ['RandomPolicy','RLSOptimalPolicy','RLSFixedOnePolicy','RLSOptimalDiscretePolicy']:
        env = make_env(bench_params)        
        policy = globals()[args.policy](env)
    elif args.policy == 'DQNPolicy':
        # get path to the trained agent model
        agent_model_file = args.policy_params
        assert agent_model_file and os.path.isfile(agent_model_file) and agent_model_file.endswith(".pt")
        # get observation space of the trained model
        train_bench_file = os.path.dirname(agent_model_file) + "/config.yml"
        _, train_bench_params, _, _, _ = read_config(train_bench_file)
        assert "observation_description" in train_bench_params
        bench_params["observation_description"] = train_bench_params["observation_description"]
        # create env
        env = make_env(bench_params)
        # create policy
        policy = DQNPolicy(env, agent_model_file)
    else:
        print(f"ERROR: policy {args.policy} not supported")
        sys.exit(1)
        
    assert env
    
    n = args.n_runs
    assert n>=1
        
    infos = []
    for i in range(n):        
        env.seed(i)
        s = env.reset()
        while True:
            act = policy.get_next_action(s)
            s, r, done, info = env.step(act)                                
            if done:                
                infos.append(info)
                break
        #print(f"Run {i}: n_evals={env.total_evals}")
        print(env.total_evals)        
                
    with open(args.out, "wb") as f:
        pickle.dump(infos, f)

main()                            