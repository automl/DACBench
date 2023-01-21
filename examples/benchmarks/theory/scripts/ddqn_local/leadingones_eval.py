import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import gym
import numpy as np

import sys
curDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(curDir)
scriptDir = os.path.dirname(curDir)
sys.path.append(scriptDir)

from calculate_optimal_policy.run import calculate_optimal_policy
from runtime_calculation import expected_run_time, variance_in_run_time

class LeadingOnesEval():
    def __init__(
        self,
        eval_env,
        agent, 
        use_formula: bool=True, # use runtime_calculation script instead of running eval_env
        n_eval_episodes_per_instance: int = 5,
        log_path: Optional[str] = None,
        save_agent_at_every_eval: bool = True,
        verbose: int = 1,
    ):
        self.best_mean_runtime = np.inf
        self.last_mean_runtime = np.inf
        self.eval_env = eval_env
        self.verbose = verbose
        self.agent = agent
        
        #self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path

        #if self.best_model_save_path is not None:
        #    os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        self.use_formula = use_formula
        self.n_eval_episodes_per_instance = n_eval_episodes_per_instance
        self.save_agent_at_every_eval = save_agent_at_every_eval
        
        # we will calculate optimal policy and its runtime for each instance
        self.instance_set = eval_env.instance_set

        # list of inst_id (keys of self.instance_set)
        self.inst_ids = eval_env.instance_id_list

        # element i^th: optimal policy for instance self.inst_ids[i]
        self.optimal_policies = [] 
        self.optimal_runtime_means = []
        self.optimal_runtime_stds = []

        # evaluation timesteps
        self.eval_timesteps = []

        # element i^th: policy at self.eval_timesteps[i] and its runtime per instance (sorted by self.inst_ids)
        self.eval_policies = []
        self.eval_policies_unclipped = []
        self.eval_runtime_means = []
        self.eval_runtime_stds = []
        
        if hasattr(eval_env, "action_choices"):
            self.action_choices = eval_env.action_choices
            self.discrete_portfolio = True
        else:
            self.discrete_portfolio = False

        if self.verbose>=1:
            print("Optimal policies:")
            
        for inst_id in self.inst_ids:

            inst = self.instance_set[inst_id]
            n = inst["size"]

            # get the optimal policy
            if self.discrete_portfolio:
                portfolio = [k for k in sorted(eval_env.action_choices, reverse=True) if k<n]
                policy = calculate_optimal_policy(n, portfolio, f"{scriptDir}/calculate_optimal_policy")
            else:
                policy = [int(n/(i+1)) for i in range(n)]
            self.optimal_policies.append(policy)
            
            # calculate the runtime of the optimal policy
            runtime_mean = expected_run_time(policy, n)
            runtime_std = np.sqrt(variance_in_run_time(policy, n))
            self.optimal_runtime_means.append(runtime_mean)
            self.optimal_runtime_stds.append(runtime_std)

            if self.verbose>=1:
                print(f"\tinstance: {inst_id}. Runtime: {runtime_mean} +/- {runtime_std}")
            if self.verbose>=2:
                print("\t" + " ".join([str(v) for v in policy]))   


    def eval(self, n_steps) -> bool:
        if self.verbose>=1:
            print(f"steps: {n_steps}")
        
        self.eval_timesteps.append(n_steps)
        
        policies = []
        policies_unclipped = []
        runtime_means = []
        runtime_stds = []

        for inst_id in self.inst_ids:
            inst = self.instance_set[inst_id]
            n = inst["size"]

            # get current policy on this instance
            policy_unclipped = [self.agent.predict(x=np.asarray([n,fx])) #TODO: only works for observation space [n,fx]
                                    for fx in range(n)]
            if self.discrete_portfolio:
                policy_unclipped = [self.action_choices[v] for v in policy_unclipped]
            policy = [np.clip(v,1,n) for v in policy_unclipped]
            policies.append(policy)
            policies_unclipped.append(policy_unclipped)

            # calculate runtime of current policy
            if self.use_formula:
                runtime_mean = expected_run_time(policy, n)
                runtime_std = np.sqrt(variance_in_run_time(policy, n))
            else:
                print("Error: not yet implemented")
                sys.exit(1)
                # # set self.eval_env's instance_set to a single instance (inst_id)
                # self.eval_env.set_attr("instance_id_list",[inst_id])
                # self.eval_env.set_attr("instance_index", 0)
                # self.eval_env.set_attr("instance_set", {inst_id: inst})
                # # evaluate on the current instance (inst_id)
                # episode_rewards, episode_lengths = evaluate_policy(
                #     self.model,
                #     self.eval_env,
                #     n_eval_episodes=self.n_eval_episodes_per_instance,
                #     render=self.render,
                #     deterministic=self.deterministic,
                #     return_episode_rewards=True,
                #     warn=self.warn,
                #     callback=self._log_success_callback,
                # )
                # # set self.eval_env's instance_set back to its original values
                # self.eval_env.set_attr("instance_id_list", self.inst_ids)
                # self.eval_env.set_attr("instance_set", self.instance_set)
                # # calculate runtime mean/std
                # runtime_mean = np.abs(np.mean(episode_rewards))
                # runtime_std = np.abs(std(episode_rewards))
            
            runtime_means.append(runtime_mean)
            runtime_stds.append(runtime_std)
            
            if self.verbose>=1:
                print(f"\tinstance: {inst_id}. Runtime: {runtime_mean} +/- {runtime_std}")
            if self.verbose>=2:
                print("\t" + " ".join([str(v) for v in policy]))

        if self.log_path is not None:                
            # save eval statistics
            self.eval_policies.append(policies)
            self.eval_runtime_means.append(runtime_means)
            self.eval_runtime_stds.append(runtime_stds)
            np.savez(self.log_path,
                    inst_ids=self.inst_ids,
                    optimal_policies=self.optimal_policies,
                    optimal_runtime_means=self.optimal_runtime_means,
                    optimal_runtime_stds=self.optimal_runtime_stds,
                    eval_timesteps=self.eval_timesteps,
                    eval_policies=self.eval_policies,
                    eval_policies_unclipped=self.eval_policies_unclipped,
                    eval_runtime_means=self.eval_runtime_means,
                    eval_runtime_stds=self.eval_runtime_stds,
                    instance_set=self.instance_set)
            # save current model
            if self.save_agent_at_every_eval:
                #self.agent.save_model(f"{os.path.dirname(self.log_path)}/model_{n_steps}")
                self.agent.save_model(f"model_{n_steps}")

        # update best_mean_runtime
        self.last_mean_runtime = runtime_mean
        if runtime_mean < self.best_mean_runtime:
            if self.verbose >= 1:
                print("New best mean runtime!")
            self.agent.save_model("best_model")
            #if self.best_model_save_path is not None:
                #self.agent.save_model(os.path.join(self.best_model_save_path, "best_model"))
                #print(self.model.__dict__)
                #print(DQN.load(os.path.join(self.best_model_save_path, "best_model"))) #DEBUG
            self.best_mean_runtime = runtime_mean
        