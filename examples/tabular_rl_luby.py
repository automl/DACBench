"""Code adapted from
"Dynamic Algorithm Configuration:Foundation of a New Meta-Algorithmic Framework"
by A. Biedenkapp and H. F. Bozkurt and T. Eimer and F. Hutter and M. Lindauer.
Original environment authors: André Biedenkapp, H. Furkan Bozkurt
"""

import sys

import numpy as np
from dacbench.benchmarks import LubyBenchmark

# Make Luby environment
from examples.example_utils import (
    EpisodeStats,
    QTable,
    get_decay_schedule,
    make_tabular_policy,
    update,
)


def q_learning(
    environment,
    num_episodes: int,
    discount_factor: float = 1.0,
    alpha: float = 0.5,
    epsilon: float = 0.1,
    verbose: bool = False,
    track_test_stats: bool = False,
    float_state=False,
    epsilon_decay: str = "const",
    decay_starts: int = 0,
    number_of_evaluations: int = 1,
    test_environment=None,
):
    """Q-Learning algorithm"""
    assert 0 <= discount_factor <= 1, "Lambda should be in [0, 1]"
    assert 0 <= epsilon <= 1, "epsilon has to be in [0, 1]"
    assert alpha > 0, "Learning rate has to be positive"
    # The action-value function.
    # Nested dict that maps state -> (action -> action-value).
    Q = QTable(environment.action_space.n, float_state)
    test_stats = None
    if track_test_stats:
        test_stats = EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes),
            expected_rewards=np.zeros(num_episodes),
        )

    # Keeps track of episode lengths and rewards
    train_stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        expected_rewards=np.zeros(num_episodes),
    )

    epsilon_schedule = get_decay_schedule(
        epsilon, decay_starts, num_episodes, epsilon_decay
    )
    for i_episode in range(num_episodes):
        epsilon = epsilon_schedule[i_episode]
        # The policy we're following
        policy = make_tabular_policy(Q, epsilon, environment.action_space.n)
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            if verbose:
                print(f"\rEpisode {i_episode + 1:>5d}/{num_episodes}.")
            else:
                print(f"\rEpisode {i_episode + 1:>5d}/{num_episodes}.", end="")
                sys.stdout.flush()
        Q, rs, exp_rew, ep_len = update(Q, environment, policy, alpha, discount_factor)
        train_stats.episode_rewards[i_episode] = rs
        train_stats.expected_rewards[i_episode] = exp_rew
        train_stats.episode_lengths[i_episode] = ep_len
    if not verbose:
        print(f"\rEpisode {i_episode + 1:>5d}/{num_episodes}.")

    return Q, (test_stats, train_stats)


bench = LubyBenchmark()
env = bench.get_environment()

# Execute 10 episodes of tabular Q-Learning
q_func, test_train_stats = q_learning(env, 10)
print(f"Rewards: {test_train_stats[1].episode_rewards}")
print(f"Episode Lenghts: {test_train_stats[1].episode_lengths}")
