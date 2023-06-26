import jax
import gymnax
import numpy as np
import jax.numpy as jnp
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import gymnasium as gym
from gymnasium.wrappers import AutoResetWrapper, FlattenObservation
import chex
from typing import Union
from gymnax.environments import EnvState, EnvParams
from flax.training.train_state import TrainState


class ExtendedTrainState(TrainState):
    @classmethod
    def create_with_opt_state(cls, *, apply_fn, params, tx, opt_state, **kwargs):
        if opt_state is None:
            opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )


def make_env(instance):
    if instance["env_framework"] == "gymnax":
        env, env_params = gymnax.make(instance["env_name"])
        env = FlattenObservationWrapper(env)
    else:
        if instance["env_name"].startswith("procgen"):
            import procgen
            import gym as old_gym
            env = old_gym.make(instance["env_name"])
            env = GymToGymnasiumWrapper(env)
        else:
            env = gym.make(instance["env_name"])
        # Gymnax does autoreset anyway
        env = AutoResetWrapper(env)
        env = FlattenObservation(env)
        env = GymToGymnaxWrapper(env)
        env_params = None
    env = LogWrapper(env)
    return env, env_params


def to_gymnasium_space(space):
    import gym as old_gym
    if isinstance(space, old_gym.spaces.Box):
        new_space = gym.spaces.Box(low=space.low, high=space.high, dtype=space.low.dtype)
    elif isinstance(space, old_gym.spaces.Discrete):
        new_space = gym.spaces.Discrete(space.n)
    else:
        raise NotImplementedError
    return new_space


class GymToGymnasiumWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = to_gymnasium_space(self.env.action_space)
        self.observation_space = to_gymnasium_space(self.env.observation_space)

    def reset(self, seed=None, options={}):
        return self.env.reset(), {}
    
    def step(self, action):
        s, r, d, i = self.env.step(action)
        return s, r, d, False, i


class JaxifyGymOutput(gym.Wrapper):
    def step(self, action):
        s, r, te, tr, _ = self.env.step(action)
        r = np.ones(s.shape) * r
        d = np.ones(s.shape) * int(te or tr)
        return np.stack([s, r, d]).astype(np.float32)


def make_bool(data):
    return np.array([bool(data)])


class GymToGymnaxWrapper(gymnax.environments.environment.Environment):
    def __init__(self, env):
        super().__init__()
        self.done = False
        self.env = JaxifyGymOutput(env)
        self.state = None
        self.state_type = None

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: EnvParams,
    ):
        """Environment-specific step transition."""
        # TODO: this is obviously super wasteful for large state spaces - how can we fix it?
        result_shape = jax.core.ShapedArray(
            np.repeat(self.state[None, ...], 3, axis=0).shape, jnp.float32
        )
        result = jax.pure_callback(self.env.step, result_shape, action)
        s = result[0].astype(self.state_type)
        r = result[1].mean()
        d = result[2].mean()
        result_shape = jax.core.ShapedArray((1,), bool)
        self.done = jax.pure_callback(make_bool, result_shape, d)[0]
        return s, {}, r, self.done, {}

    def reset_env(self, key: chex.PRNGKey, params: EnvParams):
        """Environment-specific reset."""
        self.done = False
        self.state, _ = self.env.reset()
        self.state_type = self.state.dtype
        return self.state, {}

    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state."""
        return state

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state transition is terminal."""
        return self.done

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        if isinstance(self.env.action_space, gym.spaces.Box):
            return len(self.env.action_space.low)
        else:
            return self.env.action_space.n

    def action_space(self, params: EnvParams):
        """Action space of the environment."""
        return self.env.action_space

    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        return gymnax.environments.spaces.Box(
            self.env.observation_space.low,
            self.env.observation_space.high,
            self.env.observation_space.low.shape,
        )

    def state_space(self, params: EnvParams):
        """State space of the environment."""
        return gymnax.environments.spaces.Dict({})

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(500)


def make_eval(config, network):
    env, env_params = make_env(config)

    def _env_episode(rng, env_params, network_params, _):
        reset_rng = jax.random.split(rng, 1)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        r = 0
        done = False
        while not done:
            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            #TODO: make this pretty
            try:
                pi, value = network.apply(network_params, obsv)
                action = pi.sample(seed=_rng)
            except:
                q_values = network.apply(network_params, obsv)
                action = q_values.argmax(axis=-1)
            
            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, 1)
            obsv, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(rng_step, env_state, action, env_params)
            r += reward
        return r

    def eval(rng, network_params):
        rewards = jax.vmap(_env_episode, in_axes=(None, None, None, 0))(
            rng, env_params, network_params, np.arange(config["num_eval_episodes"])
        )
        # TODO: vmap this
        # for _ in range(config["num_eval_episodes"]):
        #    rewards.append(_env_episode(rng, env_params, network_params))
        return np.mean(rewards)

    return eval