from dacbench.benchmarks import TheoryBenchmark
from stable_baselines3 import PPO
import sys
import os
sys.path.append(os.path.dirname(__file__))
from scripts import rls_policies
from rls_policies import RandomPolicy, RLSOptimalPolicy, RLSFixedOnePolicy, DQNPolicy, RLSOptimalDiscretePolicy
from ddqn_local.ddqn import DQN
import shutil
import numpy as np


def example_01():
    """
    Example 1: run RLS optimal policy with n=50 and different initial solutions
    """       
    print("\n###  Example 1: run RLSEnv optimal policy with n=50 and different initial solutions")

    # benchmark configuration
    bench_config = {"instance_set_path":"lo_rls_50.csv",
                    "discrete_action": False,
                    "min_action": 1, 
                    "max_action": 49}                    
    
    bench = TheoryBenchmark(config=bench_config)
    env = bench.get_environment(test_env=True)
    agent = RLSOptimalPolicy(env) # run optimal policy
    
    for i in range(len(env.instance_set)):
        s = env.reset()
        print(f"\nproblem size: {env.n}, initial solution: {env.x.fitness}")
        while True:
            r = agent.get_next_action(s)            
            s, rw, d, info = env.step(r)
            print(f"f(x): {env.x.fitness:3d}, #evaluations: {env.total_evals}", end="\r")
            if d:
                print("")
                break


def example_02():
    """
    Example 2: run RLS discrete power of 2 optimal policy with n=50, portfolio size of 4, and different initial solutions
    """
    print("\n###  Example 2: run RLSEnvDiscrete power of 2 optimal policy with n=50, portfolio size of 4, and different initial solutions")

    # benchmark configuration
    bench_config = {"instance_set_path":"lo_rls_50.csv", 
                    "action_choices": [1,2,4,8]}                    
    
    bench = TheoryBenchmark(config=bench_config)
    env = bench.get_environment(test_env=True)
    agent = RLSOptimalDiscretePolicy(env) # run optimal policy for discrete portfolio
    
    for i in range(len(env.instance_set)):
        s = env.reset()
        print(f"\nproblem size: {env.n}, initial solution: {env.x.fitness}")
        while True:
            r = agent.get_next_action(s)            
            s, rw, d, info = env.step(r)
            print(f"f(x): {env.x.fitness:3d}, #evaluations: {env.total_evals}", end="\r")
            if d:
                print("")
                break


def example_03():
    """
    Example 3: run RLS discrete evenly_spread random policy with n=50, portfolio size of 3 and random initial solution
    """    
    print("Example 3: run RLS discrete evenly_spread random policy with n=50, portfolio size of 3 and random initial solution")

    # benchmark configuration
    bench_config = {"instance_set_path":"lo_rls_50_random.csv", 
                    "action_choices": [1,17,33],
                    "cutoff": 1e5, # random policy can sometime perform badly so we put a reasonable cutoff here to be safe.
                    }                    
    
    bench = TheoryBenchmark(config=bench_config)
    env = bench.get_environment(test_env=True)
    agent = RandomPolicy(env)
    
    for i in range(len(env.instance_set)):
        s = env.reset()
        print(f"\nproblem size: {env.n}, initial solution: {env.x.fitness}")
        while True:
            r = agent.get_next_action(s)            
            s, rw, d, info = env.step(r)
            print(f"f(x): {env.x.fitness:3d}, #evaluations: {env.total_evals}", end="\r")
            if d:
                print("")
                break


def example_04():
    """
    Example 4: train a DDQN agent for RLS discrete power of 2 policy with n=50, portfolio size of 4 and initial solution of 40. Get the final trained agent and compare with the optimal policy for the same scenario.
    """    
    print("Example 4: train a DDQN agent for RLS discrete power of 2 policy with n=50, portfolio size of 4 and initial solution of 40. Get the final trained agent and compare with the optimal policy for the same scenario.")

    # benchmark configuration
    bench_config = {"instance_set_path":"lo_rls_50_easy.csv", 
                    "action_choices": [1,2,4,8]}
    
    bench = TheoryBenchmark(config=bench_config)
    env = bench.get_environment(test_env=False)

    # initialise DQN agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    gamma = 0.995
    out_dir = "./ddqn-output"
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    agent = DQN(state_dim, action_dim, gamma, env=env, eval_env=env, out_dir=out_dir)

    # start the training
    print("\nTraining a DDQN agent...")
    agent.train(episodes=100, eval_every_n_steps=1000, begin_learning_after=1024, batch_size=1024, save_model_interval=1000) 

    # save the final model 
    print(f"Training is complete. Saving the final model to {out_dir}/final.pt")
    agent.save_model('final')

    # create an evaluation environment
    eval_env = bench.get_environment(test_env=True)

    # evaluate the final agent
    print("\nEvaluating the final DDQN agent...")
    eval_agent = DQNPolicy(env=eval_env, model=f"{out_dir}/best.pt")
    nRuns = 500
    ddqn_results = []
    for i in range(nRuns):
        s = eval_env.reset()
        while True:
            r = eval_agent.get_next_action(s)
            s, rw, d, info = eval_env.step(r)
            if d:
                #print(f"ddqn eval run {i:3d}, optimal: {eval_env.x.is_optimal()}, evals: {eval_env.total_evals}")
                ddqn_results.append({"optimal": eval_env.x.is_optimal(), "evals": eval_env.total_evals})
                break
    n_optimals = sum([rs["optimal"] for rs in ddqn_results])
    ls_evals = np.asarray([rs["evals"] for rs in ddqn_results])
    print(f"#runs that reach optimal: {n_optimals}/{nRuns}")
    print(f"Running time (#evaluations): mean {np.mean(ls_evals):.3f}, std {np.std(ls_evals):.3f}")

    # compare with the optimal policy for the same scenario
    print("\nEvaluating the optimal policy")
    optimal_policy = RLSOptimalDiscretePolicy(eval_env)
    opt_results = []
    for i in range(nRuns):
        s = eval_env.reset()
        while True:
            r = optimal_policy.get_next_action(s)
            s, rw, d, info = eval_env.step(r)
            if d:
                opt_results.append({"optimal": eval_env.x.is_optimal(), "evals": eval_env.total_evals})
                break
    n_optimals = sum([rs["optimal"] for rs in opt_results])
    ls_evals = [rs["evals"] for rs in opt_results]    
    print(f"#runs that reach optimal: {n_optimals}/{nRuns}")
    print(f"Running time (#evaluations): mean {np.mean(ls_evals):.3f}, std {np.std(ls_evals):.3f}")


example_01()
example_02()
example_03()
example_04()
