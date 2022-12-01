import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import gym
import numpy as np

from calculate_optimal_policy.run import calculate_optimal_policy
from runtime_calculation import expected_run_time, variance_in_run_time

import sys
scriptDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(scriptDir)

from stable_baselines3 import DQN, PPO

try:
    from tqdm import TqdmExperimentalWarning

    # Remove experimental warning
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    from tqdm.rich import tqdm
except ImportError:
    # Rich not installed, we only throw an error
    # if the progress bar is used
    tqdm = None

from stable_baselines3.common import base_class  # pytype: disable=pyi-error
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization

from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.evaluation import evaluate_policy

class LeadingOnesEvalCallback(EventCallback):
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        use_formula: bool=True, # use runtime_calculation script instead of running eval_env
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes_per_instance: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        save_agent_at_every_eval: bool = True
    ):
        # from sb3's EvalCallback
        super().__init__(callback_after_eval, verbose=verbose)
        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self 
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.eval_env = eval_env
        if not use_formula:
            # Convert to VecEnv for consistency
            if not isinstance(eval_env, VecEnv):
                self.eval_env = DummyVecEnv([lambda: eval_env])
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []
        
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

        scriptDir = os.path.dirname(os.path.realpath(__file__))
        
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


    def _on_step(self) -> bool:
        continue_training = True
 
        if self.eval_freq>0 and self.n_calls%self.eval_freq==0:
            if self.verbose>=1:
                print(f"steps: {self.n_calls}")
           
            self.eval_timesteps.append(self.n_calls)
            
            policies = []
            policies_unclipped = []
            runtime_means = []
            runtime_stds = []

            for inst_id in self.inst_ids:
                inst = self.instance_set[inst_id]
                n = inst["size"]

                # get current policy on this instance
                policy_unclipped = [self.model.predict(np.asarray([n,fx]), deterministic=True)[0] #TODO: only works for observation space [n,fx]
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
                    # set self.eval_env's instance_set to a single instance (inst_id)
                    self.eval_env.set_attr("instance_id_list",[inst_id])
                    self.eval_env.set_attr("instance_index", 0)
                    self.eval_env.set_attr("instance_set", {inst_id: inst})
                    # evaluate on the current instance (inst_id)
                    episode_rewards, episode_lengths = evaluate_policy(
                        self.model,
                        self.eval_env,
                        n_eval_episodes=self.n_eval_episodes_per_instance,
                        render=self.render,
                        deterministic=self.deterministic,
                        return_episode_rewards=True,
                        warn=self.warn,
                        callback=self._log_success_callback,
                    )
                    # set self.eval_env's instance_set back to its original values
                    self.eval_env.set_attr("instance_id_list", self.inst_ids)
                    self.eval_env.set_attr("instance_set", self.instance_set)
                    # calculate runtime mean/std
                    runtime_mean = np.mean(episode_rewards)
                    runtime_std = np.std(episode_rewards)
                
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
                    self.model.save(f"{os.path.dirname(self.log_path)}/model_{self.n_calls}")

            # update mean_reward
            self.last_mean_reward = runtime_mean
            if runtime_mean > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = runtime_mean
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()
            
            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()
        
        return continue_training


    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)
