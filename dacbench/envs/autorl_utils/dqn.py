# The PPO Code is heavily based on PureJax: https://github.com/luchris429/purejaxrl
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from typing import NamedTuple


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    q_pred: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train_dqn(config, env, network):
    def train(rng, env_params, network_params, target_params, obsv, env_state):

        train_state_kwargs = {"apply_fn": network.apply, "params": network_params, "tx": optax.adam(config["lr"], eps=1e-5)}
        if config["target"]:
            train_state_kwargs["target_params"] = target_params
        train_state = TrainState.create(**train_state_kwargs)
        
        pos = 0
        observations = jnp.zeros((int(config["buffer_size"]), config["num_envs"], *env.observation_space(env_params).shape))
        next_observations = jnp.zeros((int(config["buffer_size"]), config["num_envs"], *env.observation_space(env_params).shape))
        actions = jnp.zeros((int(config["buffer_size"]), config["num_envs"], *env.action_space(env_params).shape,))
        rewards = jnp.zeros((int(config["buffer_size"]), config["num_envs"]), dtype=jnp.float32)
        dones = jnp.zeros((int(config["buffer_size"]), config["num_envs"]), dtype=jnp.float32)

        global_step = 0
        # TRAIN LOOP
        def update(train_state, observations, actions, next_observations, rewards, dones):
            if config["target"]:
                q_next_target = network.apply(train_state.target_params, next_observations)  # (batch_size, num_actions)
            else:
                q_next_target = network.apply(train_state.params, next_observations)  # (batch_size, num_actions)
            q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
            next_q_value = rewards + (1 - dones) * config["gamma"] * q_next_target

            def mse_loss(params):
                q_pred = network.apply(params, observations)  # (batch_size, num_actions)
                q_pred = q_pred[jnp.arange(q_pred.shape[0]), actions.squeeze().astype(int)]  # (batch_size,)
                return ((q_pred - next_q_value) ** 2).mean(), q_pred

            params = train_state.params
            (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss_value, q_pred, grads, params
    
        def _update_step(runner_state, unused):
            runner_state, replay = runner_state
            train_state, env_state, global_step, last_obs, rng = runner_state
            observations, actions, next_observations, rewards, dones, pos = replay
            rng, _rng = jax.random.split(rng)

            def random_action():
                return jnp.array([env.action_space().sample(rng) for _ in range(config["num_envs"])])
            
            def greedy_action():
                q_values = network.apply(train_state.params, last_obs)
                action = q_values.argmax(axis=-1)
                return action
            
            action = jax.lax.cond(jax.random.uniform(rng) < config["epsilon"], random_action, greedy_action)
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["num_envs"])
            obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
            global_step += 1

            #TODO: this is necessary for discrete obs, needs fixing though
            #if discrete:
            #    obs = obs.reshape((config["num_envs"], *env.observation_space(env_params).shape))
            #    next_obs = next_obs.reshape((config["num_envs"], *env.observation_space(env_params).shape))

            action = action.reshape((config["num_envs"], *env.action_space(env_params).shape))
            observations = observations.at[pos].set(jnp.array(last_obs).copy())
            next_observations = next_observations.at[pos].set(jnp.array(obsv).copy())
            actions = actions.at[pos].set(jnp.array(action).copy())
            rewards = rewards.at[pos].set(jnp.array(reward).copy())
            dones = dones.at[pos].set(jnp.array(done).copy())
            pos = (pos + 1) % int(config["buffer_size"]//config["num_envs"])

            def do_update(train_state):
                batch_inds = jax.random.uniform(rng, maxval=pos, shape=(config["batch_size"],)).astype(int)
                env_indices = jax.random.uniform(rng, maxval=config["num_envs"], shape=(config["batch_size"],)).astype(int)

                batch_obs = observations[batch_inds, env_indices, :]
                batch_nos = next_observations[batch_inds, env_indices, :]
                batch_as = actions[batch_inds, env_indices]
                batch_ds = dones[batch_inds, env_indices].reshape(-1, 1)
                batch_rs = rewards[batch_inds, env_indices].reshape(-1, 1)
                train_state, loss, q_pred, grads, param_history = update(
                        train_state,
                        batch_obs,
                        batch_as,
                        batch_nos,
                        batch_rs,
                        batch_ds,
                    )
                return train_state, loss, q_pred, grads, param_history

            def dont_update(train_state):
                return train_state, ((jnp.array([0]) - jnp.array([0])) ** 2).mean(), jnp.zeros(config["batch_size"]), train_state.params, train_state.params
            
            def target_update():
                return train_state.replace(target_params=optax.incremental_update(train_state.params, train_state.target_params, config["tau"]))
            
            def dont_target_update():
                return train_state
            
            train_state, loss, q_pred, grads, param_history = jax.lax.cond((global_step > config["learning_starts"]) & (global_step % config["train_frequency"] == 0), do_update, dont_update, train_state)
            if config["target"]:
                train_state = jax.lax.cond((global_step > config["learning_starts"]) & (global_step % config["target_network_frequency"] == 0), target_update, dont_target_update)

            runner_state = (train_state, env_state, global_step, obsv, rng)
            replay = (observations, actions, next_observations, rewards, dones, pos)
            if config["track_traj"]:
                metric = (loss, grads, param_history, Transition(obs=last_obs, action=action, reward=reward, done=done, info=info, q_pred=[q_pred]), {})
            elif config["track_metrics"]:
                metric = (loss, grads, {"q_pred": [q_pred]})
            else:
                metric = None
            return (runner_state, replay), metric
        
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, global_step, obsv, _rng)
        replay = (observations, actions, next_observations, rewards, dones, pos)
        (runner_state, replay), out = jax.lax.scan(
            _update_step, (runner_state, replay), None, config["total_timesteps"]
        )
        return runner_state, out

    return train
