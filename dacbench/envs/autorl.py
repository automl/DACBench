import jax
import jax.numpy as jnp
import numpy as np
from flax.training import orbax_utils
import orbax
from dacbench import AbstractEnv
from dacbench.envs.autorl_utils import make_train_ppo, make_eval, ActorCritic, make_env


class AutoRLEnv(AbstractEnv):
    ALGORITHMS = {"ppo": make_train_ppo}

    def __init__(self, config) -> None:
        super().__init__(config)
        self.checkpoint = self.config["checkpoint"]
        self.checkpoint_dir = self.config["checkpoint_dir"]
        self.checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        self.rng = jax.random.PRNGKey(self.config.seed)
        self.episode = 0
        self.track_traj = self.config.track_trajectory

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
            self.make_train = self.ALGORITHMS[config.algorithm]
        else:
            self.make_train = make_train_ppo

    def reset(self, seed: int = None, options={}):
        super().reset_(seed)
        self.env, self.env_params = make_env(self.instance)
        self.rng, _rng = jax.random.split(self.rng)
        reset_rng = jax.random.split(_rng, self.instance["num_envs"])
        self.last_obsv, self.last_env_state = jax.vmap(
            self.env.reset, in_axes=(0, None)
        )(reset_rng, self.env_params)
        self.network = ActorCritic(
            self.env.action_space(self.env_params).n,
            activation=self.instance["activation"],
            hidden_size=self.instance["hidden_size"],
        )
        init_x = jnp.zeros(self.env.observation_space(self.env_params).shape)
        _, _rng = jax.random.split(self.rng)
        if "load" in options.keys():
            restored = self.checkpointer.restore(options["load"])
            self.network_params = restored["params"]
            self.instance = restored["config"]
        else:
            self.network_params = self.network.init(_rng, init_x)
        self.eval_func = make_eval(self.instance, self.network)
        return self.get_state(self), {}

    def step(self, action):
        self.done = super().step_()
        self.instance.update(action)
        self.instance["track_traj"] = self.track_traj
        self.instance["track_metrics"] = self.config.grad_obs
        self.train_func = jax.jit(
            self.make_train(self.instance, self.env, self.network)
        )
        runner_state, metrics = self.train_func(
            self.rng,
            self.env_params,
            self.network_params,
            self.last_obsv,
            self.last_env_state,
        )
        if self.track_traj:
            metrics, self.loss_info, self.grad_info, self.traj, self.additional_info = metrics
        elif self.config.grad_obs:
            metrics, self.loss_info, self.grad_info, self.additional_info = metrics
        self.network_params = runner_state[0].params
        self.last_obsv = runner_state[2]
        self.last_env_state = runner_state[1]
        reward = self.get_reward(self)
        if self.checkpoint:
            ckpt = {
                "config": self.instance,
                "params": self.network_params,
                "metrics:": metrics,
            }
            if self.config.grad_obs:
                ckpt["value_loss"] = jnp.concatenate(self.loss_info[0], axis=0)
                ckpt["actor_loss"] = jnp.concatenate(self.loss_info[1], axis=0)
                ckpt["gradients"] = self.grad_info["params"]
                for k in self.additional_info:
                    if k == "minibatches":
                        ckpt["minibatch"] = {}
                        ckpt["minibatch"]["states"] = jnp.concatenate(self.additional_info[k][0].obs, axis=0)
                        ckpt["minibatch"]["value"] = jnp.concatenate(self.additional_info[k][0].value, axis=0)
                        ckpt["minibatch"]["action"] = jnp.concatenate(self.additional_info[k][0].action, axis=0)
                        ckpt["minibatch"]["reward"] = jnp.concatenate(self.additional_info[k][0].reward, axis=0)
                        ckpt["minibatch"]["log_prob"] = jnp.concatenate(self.additional_info[k][0].log_prob, axis=0)
                        ckpt["minibatch"]["dones"] = jnp.concatenate(self.additional_info[k][0].done, axis=0)
                        ckpt["minibatch"]["advantages"] = jnp.concatenate(self.additional_info[k][1], axis=0)
                        ckpt["minibatch"]["targets"] = jnp.concatenate(self.additional_info[k][2], axis=0)
                    else:
                        ckpt[k] = jnp.concatenate(self.additional_info[k], axis=0)

            if self.instance["track_traj"]:
                ckpt["trajectory"] = {}
                ckpt["trajectory"]["states"] = jnp.concatenate(self.traj.obs, axis=0)
                ckpt["trajectory"]["value"] = jnp.concatenate(self.traj.value, axis=0)
                ckpt["trajectory"]["action"] = jnp.concatenate(self.traj.action, axis=0)
                ckpt["trajectory"]["reward"] = jnp.concatenate(self.traj.reward, axis=0)
                ckpt["trajectory"]["log_prob"] = jnp.concatenate(self.traj.log_prob, axis=0)
                ckpt["trajectory"]["dones"] = jnp.concatenate(self.traj.done, axis=0)
            save_args = orbax_utils.save_args_from_target(ckpt)
            checkpoint_name = self.checkpoint_dir + "/"
            if "checkpoint_name" in self.config.keys():
                checkpoint_name += self.config["checkpoint_name"]
            else:
                if not self.done:
                    checkpoint_name += f"_episode_{self.episode}_step_{self.c_step}"
                else:
                    checkpoint_name += "final"
            
            self.checkpointer.save(
                checkpoint_name,
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
            grad_info = [self.grad_info["params"][k][kk] for k in self.grad_info["params"] for kk in self.grad_info["params"][k]]
            grad_norm = np.mean([jnp.linalg.norm(g) for g in grad_info])
            grad_var = np.mean([jnp.var(g) for g in grad_info])
        return np.array([self.c_step, self.c_step * self.instance["total_timesteps"], grad_norm, grad_var])
