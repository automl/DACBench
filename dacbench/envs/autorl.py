import jax
import jax.numpy as jnp
import numpy as np
from flax.training import orbax_utils
import gymnax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import orbax
from dacbench import AbstractEnv
from dacbench.envs.autorl_utils import make_train_ppo, make_eval, ActorCritic


class AutoRLEnv(AbstractEnv):
    ALGORITHMS = {"ppo": make_train_ppo}
    def __init__(self, config) -> None:
        super().__init__(config)
        self.checkpoint = self.config["checkpoint"]
        self.checkpoint_dir = self.config["checkpoint_dir"]
        self.checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        self.rng = jax.random.PRNGKey(30)#self.config.seed)
        self.episode = 0

        if "reward_function" in config.keys():
            self.get_reward = config["reward_function"]
        else:
            self.get_reward = self.get_default_reward

        if "state_method" in config.keys():
            self.get_state = config["state_method"]
        else:
            self.get_state = self.get_default_state

        if "algorithm" in config.keys():
            self.make_train = self.ALGORITHMS[config.algorithm]
        else:
            self.make_train = make_train_ppo


    def reset(self, seed: int = None, options={}):
        super().reset_(seed)
        self.env, self.env_params = gymnax.make(self.instance["env_name"])
        # TODO: env wrapping should be optional
        # TODO: probably should use auto-reset wrapper, though
        self.env = FlattenObservationWrapper(self.env)
        self.env = LogWrapper(self.env)
        self.rng, _rng = jax.random.split(self.rng)
        reset_rng = jax.random.split(_rng, self.instance["num_envs"])
        self.last_obsv, self.last_env_state = jax.vmap(self.env.reset, in_axes=(0, None))(reset_rng, self.env_params)
        self.network = ActorCritic(self.env.action_space(self.env_params).n, activation=self.instance["activation"], hidden_size=self.instance["hidden_size"])
        init_x = jnp.zeros(self.env.observation_space(self.env_params).shape)
        _, _rng = jax.random.split(self.rng)
        if "load" in options.keys():
            restored = self.checkpointer.restore(options["load"])
            self.network_params = restored["params"]
            self.instance = restored["config"]
        else:
            self.network_params = self.network.init(_rng, init_x)
        # TODO: jit
        self.eval_func = make_eval(self.instance, self.network)
        return self.get_state(self), {}
    
    def step(self, action):
        self.done = super().step_()
        self.instance.update(action)
        self.train_func = jax.jit(self.make_train(self.instance, self.env, self.network))
        out = self.train_func(self.rng, self.env_params, self.network_params, self.last_obsv, self.last_env_state)
        self.network_params = out["runner_state"][0].params
        self.last_obsv = out["runner_state"][2]
        self.last_env_state = out["runner_state"][1]
        reward = self.get_reward(self)
        if self.checkpoint and self.done:
            ckpt = {'config': self.instance, 'params': self.network_params}
            save_args = orbax_utils.save_args_from_target(ckpt)
            self.checkpointer.save(self.checkpoint_dir+f"_episode_{self.episode}_step_{self.c_step}", ckpt, save_args=save_args)
        return self.get_state(self), reward, False, self.done, {}
    
    def get_default_reward(self, _):
        return self.eval_func(self.rng, self.network_params)

    # TODO: implement
    def get_default_state(self, _):
        return np.array([])

