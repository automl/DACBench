import jax
import jax.numpy as jnp
import coax
import haiku as hk
from numpy import prod
import optax

from dacbench.benchmarks import CMAESBenchmark
from dacbench.wrappers import ObservationWrapper


# the name of this script
name = "ppo"

# the Pendulum MDP
bench = CMAESBenchmark()
env = bench.get_environment()
env = ObservationWrapper(env)
env = coax.wrappers.TrainMonitor(
    env, name=name, tensorboard_dir=f"./data/tensorboard/{name}"
)


def func_pi(S, is_training):
    shared = hk.Sequential(
        (
            hk.Linear(8),
            jax.nn.relu,
            hk.Linear(8),
            jax.nn.relu,
        )
    )
    mu = hk.Sequential(
        (
            shared,
            hk.Linear(8),
            jax.nn.relu,
            hk.Linear(prod(env.action_space.shape), w_init=jnp.zeros),
            hk.Reshape(env.action_space.shape),
        )
    )
    logvar = hk.Sequential(
        (
            shared,
            hk.Linear(8),
            jax.nn.relu,
            hk.Linear(prod(env.action_space.shape), w_init=jnp.zeros),
            hk.Reshape(env.action_space.shape),
        )
    )
    return {"mu": mu(S), "logvar": logvar(S)}


def func_v(S, is_training):
    seq = hk.Sequential(
        (
            hk.Linear(8),
            jax.nn.relu,
            hk.Linear(8),
            jax.nn.relu,
            hk.Linear(8),
            jax.nn.relu,
            hk.Linear(1, w_init=jnp.zeros),
            jnp.ravel,
        )
    )
    return seq(S)


# define function approximators
pi = coax.Policy(func_pi, env)
v = coax.V(func_v, env)


# target network
pi_targ = pi.copy()


# experience tracer
tracer = coax.reward_tracing.NStep(n=5, gamma=0.9)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=512)


# policy regularizer (avoid premature exploitation)
policy_reg = coax.regularizers.EntropyRegularizer(pi, beta=0.01)


# updaters
simpletd = coax.td_learning.SimpleTD(v, optimizer=optax.adam(1e-3))
ppo_clip = coax.policy_objectives.PPOClip(
    pi, regularizer=policy_reg, optimizer=optax.adam(1e-4)
)

# train
for _ in range(10):
    done, truncated = False, False
    s, info = env.reset()

    while not (done or truncated):
        a, logp = pi_targ(s, return_logp=True)
        s_next, r, done, truncated, info = env.step(a)

        # trace rewards
        tracer.add(s, a, r, done or truncated, logp)
        while tracer:
            buffer.add(tracer.pop())

        # learn
        if len(buffer) >= buffer.capacity:
            for _ in range(int(4 * buffer.capacity / 32)):  # 4 passes per round
                transition_batch = buffer.sample(batch_size=32)
                metrics_v, td_error = simpletd.update(
                    transition_batch, return_td_error=True
                )
                metrics_pi = ppo_clip.update(transition_batch, td_error)
                env.record_metrics(metrics_v)
                env.record_metrics(metrics_pi)

            buffer.clear()
            pi_targ.soft_update(pi, tau=0.1)

        if done or truncated:
            break

        s = s_next
