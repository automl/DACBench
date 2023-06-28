import jax
import jax.numpy as jnp
import numpy as np
from flax.training import orbax_utils
import orbax
from dacbench import AbstractEnv
from dacbench.envs.autorl_utils import make_train_ppo, make_train_dqn, make_eval, ActorCritic, Q, make_env
import gymnax


class AutoRLEnv(AbstractEnv):
    ALGORITHMS = {"ppo": (make_train_ppo, ActorCritic), "dqn": (make_train_dqn, Q)}

    def __init__(self, config) -> None:
        super().__init__(config)
        self.checkpoint = self.config["checkpoint"]
        self.checkpoint_dir = self.config["checkpoint_dir"]
        self.rng = jax.random.PRNGKey(self.config.seed)
        self.episode = 0

        if "reward_function" in config.keys():
            self.get_reward = config["reward_function"]
        else:
            self.get_reward = self.get_default_reward

        if "state_method" in config.keys():
            self.get_state = config["state_method"]
        elif self.config.grad_obs:
            self.get_state = self.get_gradient_state
        else:
            self.get_state = self.get_default_state

        if "algorithm" in config.keys():
            self.make_train = self.ALGORITHMS[config.algorithm][0]
            self.network_cls = self.ALGORITHMS[config.algorithm][1]
        else:
            self.make_train = make_train_ppo
            self.network_cls = ActorCritic

    def reset(self, seed: int = None, options={}):
        super().reset_(seed)
        self.env, self.env_params = make_env(self.instance)
        self.rng, _rng = jax.random.split(self.rng)
        reset_rng = jax.random.split(_rng, self.instance["num_envs"])
        self.last_obsv, self.last_env_state = jax.vmap(
            self.env.reset, in_axes=(0, None)
        )(reset_rng, self.env_params)
        if isinstance(self.env.action_space(), gymnax.environments.spaces.Discrete):
            action_size = self.env.action_space(self.env_params).n
            discrete = True
        elif isinstance(self.env.action_space(), gymnax.environments.spaces.Box):
            action_size = self.env.action_space(self.env_params).shape[0]
            discrete = False
        else:
            raise NotImplementedError(f"Only Discrete and Box action spaces are supported, got {self.env.action_space()}.")
        
        self.network = self.network_cls(
            action_size,
            activation=self.instance["activation"],
            hidden_size=self.instance["hidden_size"],
            discrete=discrete,
        )

        init_x = jnp.zeros(self.env.observation_space(self.env_params).shape)
        _, _rng = jax.random.split(self.rng)
        if "load" in options.keys():
            checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            restored = checkpointer.restore(options["load"])
            self.network_params = restored["params"]
            if isinstance(self.network_params, list):
                self.network_params = self.network_params[0]
            self.instance = restored["config"]
            if "target" in restored.keys():
                self.target_params = restored["target"][0]
                if isinstance(self.target_params, list):
                    self.target_params = self.target_params[0]
            try:
                self.opt_state = restored["opt_state"]
            except:
                self.opt_state = None
        else:
            self.network_params = self.network.init(_rng, init_x)
            self.target_params = self.network.init(_rng, init_x)
            self.opt_state = None
        self.eval_func = make_eval(self.instance, self.network)
        return self.get_state(self), {}

    def step(self, action):
        self.done = super().step_()
        self.instance.update(action)
        self.instance["track_traj"] = "trajectory" in self.checkpoint
        self.instance["track_metrics"] = self.config.grad_obs
        self.train_func = jax.jit(
            self.make_train(self.instance, self.env, self.network)
        )

        train_args = (self.rng, self.env_params, self.network_params, self.opt_state, self.last_obsv, self.last_env_state)
        if self.config.algorithm == "dqn":
            train_args = (self.rng, self.env_params, self.network_params, self.target_params, self.opt_state, self.last_obsv, self.last_env_state)

        runner_state, metrics = self.train_func(*train_args)
        if "trajectory" in self.checkpoint:
            self.loss_info, self.grad_info, self.opt_info, self.traj, self.additional_info = metrics
        elif self.config.grad_obs:
            self.loss_info, self.grad_info, self.opt_info, self.additional_info = metrics
        self.network_params = runner_state[0].params
        self.last_obsv = runner_state[2]
        self.last_env_state = runner_state[1]
        reward = self.get_reward(self)
        if self.checkpoint:
            #Checkpoint setup
            checkpoint_name = self.checkpoint_dir + "/"
            if "checkpoint_name" in self.config.keys():
                checkpoint_name += self.config["checkpoint_name"]
            else:
                if not self.done:
                    checkpoint_name += f"_episode_{self.episode}_step_{self.c_step}"
                else:
                    checkpoint_name += "final"
            
            ckpt = {
                "config": self.instance,
                "optimizer_state": self.opt_state,
            }

            if "policy" in self.checkpoint:
                ckpt["params"] = self.network_params
                if "target" in self.instance.keys():
                    if self.instance["target"]:
                        ckpt["target"] = self.target_params

            save_args = orbax_utils.save_args_from_target(ckpt)
            checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            checkpointer.save(
                checkpoint_name,
                ckpt,
                save_args=save_args
            )

            if "loss" in self.checkpoint:
                ckpt = {}
                if self.config.algorithm == "ppo":
                    ckpt["value_loss"] = jnp.concatenate(self.loss_info[0], axis=0)
                    ckpt["actor_loss"] = jnp.concatenate(self.loss_info[1], axis=0)
                elif self.config.algorithm == "dqn":
                    ckpt["loss"] = self.loss_info

                save_args = orbax_utils.save_args_from_target(ckpt)
                checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                loss_checkpoint = checkpoint_name+"_loss"
                checkpointer.save(
                    loss_checkpoint,
                    ckpt,
                    save_args=save_args,
                    )
                
            if "minibatches" in self.checkpoint and "trajectory" in self.checkpoint:
                ckpt = {}
                ckpt["minibatches"] = {}
                ckpt["minibatches"]["states"] = jnp.concatenate(self.additional_info["minibatches"][0].obs, axis=0)
                ckpt["minibatches"]["value"] = jnp.concatenate(self.additional_info["minibatches"][0].value, axis=0)
                ckpt["minibatches"]["action"] = jnp.concatenate(self.additional_info["minibatches"][0].action, axis=0)
                ckpt["minibatches"]["reward"] = jnp.concatenate(self.additional_info["minibatches"][0].reward, axis=0)
                ckpt["minibatches"]["log_prob"] = jnp.concatenate(self.additional_info["minibatches"][0].log_prob, axis=0)
                ckpt["minibatches"]["dones"] = jnp.concatenate(self.additional_info["minibatches"][0].done, axis=0)
                ckpt["minibatches"]["advantages"] = jnp.concatenate(self.additional_info["minibatches"][1], axis=0)
                ckpt["minibatches"]["targets"] = jnp.concatenate(self.additional_info["minibatches"][2], axis=0)
                save_args = orbax_utils.save_args_from_target(ckpt)
                checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                minibatch_checkpoint = checkpoint_name+"_minibatches"
                checkpointer.save(
                    minibatch_checkpoint,
                    ckpt,
                    save_args=save_args,
                    )

            if "extras" in self.checkpoint:
                ckpt = {}
                for k in self.additional_info:
                    if k == "param_history":
                        ckpt[k] = self.additional_info[k]
                    elif k != "minibatches":
                        ckpt[k] = jnp.concatenate(self.additional_info[k], axis=0)
                
                save_args = orbax_utils.save_args_from_target(ckpt)
                checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                extras_checkpoint = checkpoint_name+"_extras"
                checkpointer.save(
                    extras_checkpoint,
                    ckpt,
                    save_args=save_args,
                    )

            if "trajectory" in self.checkpoint:
                ckpt = {}
                ckpt["trajectory"] = {}
                ckpt["trajectory"]["states"] = jnp.concatenate(self.traj.obs, axis=0)
                ckpt["trajectory"]["action"] = jnp.concatenate(self.traj.action, axis=0)
                ckpt["trajectory"]["reward"] = jnp.concatenate(self.traj.reward, axis=0)
                ckpt["trajectory"]["dones"] = jnp.concatenate(self.traj.done, axis=0)
                if self.config.algorithm == "ppo":
                    ckpt["trajectory"]["value"] = jnp.concatenate(self.traj.value, axis=0)
                    ckpt["trajectory"]["log_prob"] = jnp.concatenate(self.traj.log_prob, axis=0)
                elif self.config.algorithm == "dqn":
                    ckpt["trajectory"]["q_pred"] = jnp.concatenate(self.traj.q_pred, axis=0)

                save_args = orbax_utils.save_args_from_target(ckpt)
                checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                traj_checkpoint = checkpoint_name+"_trajectory"
                checkpointer.save(
                    traj_checkpoint,
                    ckpt,
                    save_args=save_args,
                    )
            
        return self.get_state(self), reward, False, self.done, {}

    def get_default_reward(self, _):
        return self.eval_func(self.rng, self.network_params)

    # Useful features could be: total deltas of grad norm and grad var, instance info...
    def get_default_state(self, _):
        return np.array([self.c_step, self.c_step * self.instance["total_timesteps"]])

    def get_gradient_state(self, _):
        if self.c_step == 0:
            grad_norm = 0
            grad_var = 0
        else:
            grad_info = self.grad_info
            if self.config.algorithm == "ppo":
                import flax
                grad_info = grad_info["params"]
                grad_info = {k: v for (k, v) in grad_info.items() if isinstance(v, flax.core.frozen_dict.FrozenDict)}
            grad_info = [grad_info[g][k] for g in grad_info.keys() for k in grad_info[g].keys()]
            grad_norm = np.mean([jnp.linalg.norm(g) for g in grad_info])
            grad_var = np.mean([jnp.var(g) for g in grad_info])
        return np.array([self.c_step, self.c_step * self.instance["total_timesteps"], grad_norm, grad_var])
