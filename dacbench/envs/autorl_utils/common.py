import jax
import gymnax
import numpy as np
import jax.numpy as jnp
from typing import Sequence
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    hidden_size: int = 64

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)
    

class Q(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        
        q = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        q = activation(q)
        q = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(q)
        q = activation(q)
        q = nn.Dense(self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            q
        )

        return q

def make_eval(config, network):
    env, env_params = gymnax.make(config["env_name"])
    # TODO: env wrapping should be optional
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    def _env_episode(rng, env_params, network_params, _):
        reset_rng = jax.random.split(rng, 1)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        r = 0
        done = False
        while not done:
            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            pi, value = network.apply(network_params, obsv)
            action = pi.sample(seed=_rng)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, 1)
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                        rng_step, env_state, action, env_params
                    )
            r += reward
        return r
    
    def eval(rng, network_params):
        rewards = jax.vmap(_env_episode, in_axes=(None, None, None, 0))(rng, env_params, network_params, np.arange(config["num_eval_episodes"]))
        #TODO: vmap this
        #for _ in range(config["num_eval_episodes"]):
        #    rewards.append(_env_episode(rng, env_params, network_params))
        return np.mean(rewards)
    return eval