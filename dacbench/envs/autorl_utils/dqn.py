# The PPO Code is heavily based on PureJax: https://github.com/luchris429/purejaxrl
import jax
import jax.numpy as jnp
import chex
import optax
from .common import ExtendedTrainState
from typing import NamedTuple
import dejax.utils as utils
from typing import Callable, Any, Tuple
import gymnax
import chex
import jax.lax

ReplayBufferState = Any
Item = chex.ArrayTree
ItemBatch = chex.ArrayTree
IntScalar = chex.Array
ItemUpdateFn = Callable[[Item], Item]
BoolScalar = chex.Array


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    q_pred: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_default_add_batch_fn(add_fn):
    def add_batch_fn(
        state: ReplayBufferState, item_batch: ItemBatch
    ) -> ReplayBufferState:
        def scan_body(state: ReplayBufferState, adds) -> Tuple[ReplayBufferState, None]:
            item, weight = adds
            state = add_fn(state, item, weight)
            return state, None

        state, _ = jax.lax.scan(f=scan_body, init=state, xs=item_batch)
        return state

    return add_batch_fn


@chex.dataclass(frozen=False)
class ReplayBuffer:
    init_fn: Callable[[Item], ReplayBufferState]
    size_fn: Callable[[ReplayBufferState], IntScalar]
    add_fn: Callable[[ReplayBufferState, Item], ReplayBufferState]
    add_batch_fn: Callable[[ReplayBufferState, ItemBatch], ReplayBufferState]
    sample_fn: Callable[[ReplayBufferState, chex.PRNGKey, int], ItemBatch]
    update_fn: Callable[[ReplayBufferState, ItemUpdateFn], ReplayBufferState]


@chex.dataclass(frozen=False)
class CircularBuffer:
    data: ItemBatch
    weights: ItemBatch
    head: IntScalar
    tail: IntScalar
    full: BoolScalar


def init(item_prototype: Item, weight_prototype, max_size: int) -> CircularBuffer:
    chex.assert_tree_has_only_ndarrays(item_prototype)

    data = jax.tree_util.tree_map(
        lambda t: utils.tile_over_axis(t, axis=0, size=max_size), item_prototype
    )
    weights = jax.tree_util.tree_map(
        lambda t: utils.tile_over_axis(t, axis=0, size=max_size), weight_prototype
    )
    return CircularBuffer(
        data=data,
        weights=weights,
        head=utils.scalar_to_jax(0),
        tail=utils.scalar_to_jax(0),
        full=utils.scalar_to_jax(False),
    )


def max_size(buffer: CircularBuffer) -> int:
    return utils.get_pytree_axis_dim(buffer.data, axis=0)


def size(buffer: CircularBuffer) -> IntScalar:
    return jax.lax.select(
        buffer.full,
        on_true=max_size(buffer),
        on_false=jax.lax.select(
            buffer.head >= buffer.tail,
            on_true=buffer.head - buffer.tail,
            on_false=max_size(buffer) - (buffer.tail - buffer.head),
        ),
    )


def push(buffer: CircularBuffer, item: Item, weight: Item) -> CircularBuffer:
    chex.assert_tree_has_only_ndarrays(item)

    insert_pos = buffer.head
    new_data = utils.set_pytree_batch_item(buffer.data, insert_pos, item)
    new_weights = utils.set_pytree_batch_item(buffer.weights, insert_pos, weight)
    new_head = (insert_pos + 1) % max_size(buffer)
    new_tail = jax.lax.select(
        buffer.full,
        on_true=new_head,
        on_false=buffer.tail,
    )
    new_full = new_head == new_tail

    return buffer.replace(
        data=new_data, head=new_head, tail=new_tail, full=new_full, weights=new_weights
    )


def pop(buffer: CircularBuffer) -> (Item, CircularBuffer):
    remove_pos = buffer.tail
    popped_item = utils.get_pytree_batch_item(buffer.data, remove_pos)
    new_tail = (remove_pos + 1) % max_size(buffer)
    new_full = utils.scalar_to_jax(False)

    return popped_item, buffer.replace(tail=new_tail, full=new_full)


def get_at_index(buffer: CircularBuffer, index: IntScalar) -> Item:
    chex.assert_shape(index, ())
    index = (buffer.tail + index) % max_size(buffer)
    return utils.get_pytree_batch_item(buffer.data, index), utils.get_pytree_batch_item(
        buffer.weights, index
    )


def uniform_sample(
    buffer: CircularBuffer, rng: chex.PRNGKey, batch_size: int
) -> ItemBatch:
    sample_pos = jax.random.randint(
        rng, minval=0, maxval=size(buffer.storage), shape=(batch_size,)
    )
    get_at_index_batch = jax.vmap(get_at_index, in_axes=(None, 0))
    transition_batch, _ = get_at_index_batch(buffer.storage, sample_pos)
    return transition_batch


def weighted_sample(
    buffer: CircularBuffer, rng: chex.PRNGKey, batch_size: int
) -> ItemBatch:
    sample_pos = jax.random.randint(
        rng, minval=0, maxval=size(buffer.storage), shape=(batch_size,)
    )
    get_at_index_batch = jax.vmap(get_at_index, in_axes=(None, 0))
    transition_batch, P = get_at_index_batch(buffer.storage, sample_pos)
    P_scaled = P / sum(buffer.storage.weights)  # prioritized, biased propensities
    W = jnp.power(
        P_scaled * size(buffer.storage), buffer.beta
    )  # inverse propensity weights (β≈1)
    W /= (
        W.max()
    )  # for stability, ensure only down-weighting (see sec. 3.4 of arxiv:1511.05952)
    updated_weights = P * W
    for idx, w in zip(sample_pos, updated_weights):
        P = utils.set_pytree_batch_item(buffer.storage.weights, idx, w)
    buffer.weights = buffer.storage.replace(weights=P)
    return transition_batch


@chex.dataclass(frozen=False)
class UniformReplayBufferState:
    storage: CircularBuffer
    beta: float


def uniform_replay(max_size: int, beta: float):
    def init_fn(beta, item_prototype, weight_prototype) -> UniformReplayBufferState:
        return UniformReplayBufferState(
            storage=init(item_prototype, weight_prototype, max_size), beta=beta
        )

    def size_fn(state: UniformReplayBufferState):
        return size(state.storage)

    def add_fn(
        state: UniformReplayBufferState, item, weight
    ) -> UniformReplayBufferState:
        return state.replace(storage=push(state.storage, item, weight))

    def sample_fn(state: UniformReplayBufferState, rng: chex.PRNGKey, batch_size: int):
        return uniform_sample(state.storage, rng, batch_size)

    def update_fn(
        state: UniformReplayBufferState, item_update_fn
    ) -> UniformReplayBufferState:
        # TODO: there might be a faster way to make updates that does not affect all items in the buffer
        batch_update_fn = jax.vmap(item_update_fn)
        updated_data = batch_update_fn(state.storage.data)
        return state.replace(storage=state.storage.replace(data=updated_data))

    return ReplayBuffer(
        init_fn=jax.tree_util.Partial(init_fn, beta),
        size_fn=jax.tree_util.Partial(size_fn),
        add_fn=jax.tree_util.Partial(add_fn),
        # TODO: it should be possible to make an optimized version of add_batch_fn for this buffer type
        add_batch_fn=jax.tree_util.Partial(make_default_add_batch_fn(add_fn)),
        sample_fn=jax.tree_util.Partial(sample_fn),
        update_fn=jax.tree_util.Partial(update_fn),
    )


def make_train_dqn(config, env, network, _):
    def train(
        rng,
        env_params,
        network_params,
        target_params,
        opt_state,
        obsv,
        env_state,
        buffer_state,
        global_step
    ):
        train_state_kwargs = {
            "apply_fn": network.apply,
            "params": network_params,
            "tx": optax.adam(config["lr"], eps=1e-5),
            "opt_state": opt_state,
        }
        train_state_kwargs["target_params"] = target_params
        train_state = ExtendedTrainState.create_with_opt_state(**train_state_kwargs)
        buffer = uniform_replay(
            max_size=int(config["buffer_size"]), beta=config["beta"]
        )
        buffer.sample_fn = (
            weighted_sample if config["prioritize_replay"] else uniform_sample
        )
        global_step = global_step

        # TRAIN LOOP
        def update(
            train_state, observations, actions, next_observations, rewards, dones
        ):
            if config["target"]:
                q_next_target = network.apply(
                    train_state.target_params, next_observations
                )  # (batch_size, num_actions)
            else:
                q_next_target = network.apply(
                    train_state.params, next_observations
                )  # (batch_size, num_actions)
            q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
            next_q_value = rewards + (1 - dones) * config["gamma"] * q_next_target

            def mse_loss(params):
                q_pred = network.apply(
                    params, observations
                )  # (batch_size, num_actions)
                q_pred = q_pred[
                    jnp.arange(q_pred.shape[0]), actions.squeeze().astype(int)
                ]  # (batch_size,)
                return ((q_pred - next_q_value) ** 2).mean(), q_pred

            (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(
                train_state.params
            )
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss_value, q_pred, grads, train_state.opt_state

        def _update_step(runner_state, unused):
            (
                train_state,
                env_state,
                last_obs,
                rng,
                buffer_state,
                global_step,
            ) = runner_state
            rng, _rng = jax.random.split(rng)

            def random_action():
                return jnp.array(
                    [
                        env.action_space(env_params).sample(rng)
                        for _ in range(config["num_envs"])
                    ]
                )

            def greedy_action():
                q_values = network.apply(train_state.params, last_obs)
                action = q_values.argmax(axis=-1)
                return action

            def take_step(carry, _):
                obsv, env_state, global_step, buffer_state = carry
                action = jax.lax.cond(
                    jax.random.uniform(rng) < config["epsilon"],
                    random_action,
                    greedy_action,
                )

                rng_step = jax.random.split(_rng, config["num_envs"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                action = jnp.expand_dims(action, -1)
                done = jnp.expand_dims(done, -1)
                reward = jnp.expand_dims(reward, -1)

                def no_target_td(train_state):
                    return network.apply(train_state.params, obsv).argmax(axis=-1)

                def target_td(train_state):
                    return network.apply(train_state.target_params, obsv).argmax(
                        axis=-1
                    )

                q_next_target = jax.lax.cond(
                    config["target"], target_td, no_target_td, train_state
                )

                td_error = (
                    reward
                    + (1 - done) * config["gamma"] * jnp.expand_dims(q_next_target, -1)
                    - network.apply(train_state.params, last_obs).take(action)
                )
                transition_weight = jnp.power(
                    jnp.abs(td_error) + config["buffer_epsilon"], config["alpha"]
                )

                buffer_state = buffer.add_batch_fn(
                    buffer_state,
                    ((last_obs, obsv, action, reward, done), transition_weight),
                )
                global_step += 1
                return (obsv, env_state, global_step, buffer_state), (
                    obsv,
                    action,
                    reward,
                    done,
                    info,
                    td_error,
                )

            def do_update(train_state, buffer_state):
                batch = buffer.sample_fn(buffer_state, rng, config["batch_size"])
                train_state, loss, q_pred, grads, opt_state = update(
                    train_state,
                    batch[0],
                    batch[2],
                    batch[1],
                    batch[3],
                    batch[4],
                )
                return train_state, loss, q_pred, grads, opt_state

            def dont_update(train_state, _):
                return (
                    train_state,
                    ((jnp.array([0]) - jnp.array([0])) ** 2).mean(),
                    jnp.ones(config["batch_size"]),
                    train_state.params,
                    train_state.opt_state,
                )

            def target_update():
                return train_state.replace(
                    target_params=optax.incremental_update(
                        train_state.params, train_state.target_params, config["tau"]
                    )
                )

            def dont_target_update():
                return train_state

            (last_obs, env_state, global_step, buffer_state), (
                observations,
                action,
                reward,
                done,
                info,
                td_error,
            ) = jax.lax.scan(
                take_step,
                (last_obs, env_state, global_step, buffer_state),
                None,
                config["train_frequency"],
            )

            train_state, loss, q_pred, grads, opt_state = jax.lax.cond(
                (global_step > config["learning_starts"])
                & (global_step % config["train_frequency"] == 0),
                do_update,
                dont_update,
                train_state,
                buffer_state,
            )
            train_state = jax.lax.cond(
                (global_step > config["learning_starts"])
                & (global_step % config["target_network_update_freq"] == 0),
                target_update,
                dont_target_update,
            )
            runner_state = (
                train_state,
                env_state,
                last_obs,
                rng,
                buffer_state,
                global_step,
            )
            if config["track_traj"]:
                metric = (
                    loss,
                    grads,
                    opt_state,
                    Transition(
                        obs=observations,
                        action=action,
                        reward=reward,
                        done=done,
                        info=info,
                        q_pred=[q_pred],
                    ),
                    {"td_error": [td_error]},
                )
            elif config["track_metrics"]:
                metric = (
                    loss,
                    grads,
                    opt_state,
                    {"q_pred": [q_pred], "td_error": [td_error]},
                )
            else:
                metric = None
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng, buffer_state, global_step)
        runner_state, out = jax.lax.scan(
            _update_step, runner_state, None, (config["total_timesteps"]//config["train_frequency"])//config["num_envs"]
        )
        return runner_state, out

    return train
