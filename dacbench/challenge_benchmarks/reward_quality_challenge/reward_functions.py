from __future__ import annotations

import numpy as np


def easy_sigmoid(self):
    sigmoids = [
        np.abs(self._sig(self.c_step, slope, shift))
        for slope, shift in zip(self.shifts, self.slopes, strict=False)
    ]
    action = []
    for i in range(len(self.action_vals)):
        best_action = None
        dist = 100
        for a in range(self.action_vals[i] + 1):
            if np.abs(sigmoids[i] - a / (self.action_vals[i] - 1)) < dist:
                dist = np.abs(sigmoids[i] - a / (self.action_vals[i]))
                best_action = a
        action.append(best_action)
    action_diffs = self.action - action
    r = 0
    for i in range(len(action_diffs)):
        r += 10**i * action_diffs[i]
    return max(self.reward_range[0], min(self.reward_range[1], r))


def almost_easy_sigmoid(self):
    r = [
        1 - np.abs(self._sig(self.c_step, slope, shift) - (act / (max_act - 1)))
        for slope, shift, act, max_act in zip(
            self.slopes, self.shifts, self.action, self.action_vals, strict=False
        )
    ]
    r = sum(r)
    return max(self.reward_range[0], min(self.reward_range[1], r))


def sum_reward(self):
    if self.c_step == 1:
        self.rew_sum = 0
    self.rew_sum += self.get_default_reward
    if self.done:
        return self.rew_sum
    else:
        return 0


def random_reward(self):
    return np.random.uniform(self.reward_range[0], self.reward_range[1])


def manhattan_distance_reward_geometric(self):
    def manhattan(a, b):
        return sum(abs(val1 - val2) for val1, val2 in zip(a, b, strict=False))

    coordinates, action_intervall, highest_coords, lowest_actions = self._pre_reward()
    manhattan_dist = manhattan(coordinates, action_intervall)

    max_dist = manhattan(lowest_actions, highest_coords)
    reward = 1 - (manhattan_dist / max_dist)

    return abs(reward)


def quadratic_manhattan_distance_reward_geometric(self):
    def manhattan(a, b):
        return sum(abs(val1 - val2) for val1, val2 in zip(a, b, strict=False))

    coordinates, action_intervall, highest_coords, lowest_actions = self._pre_reward()
    manhattan_dist = manhattan(coordinates, action_intervall)

    max_dist = manhattan(lowest_actions, highest_coords)
    reward = (1 - (manhattan_dist / max_dist)) ** 2

    return abs(reward)


def quadratic_euclidean_distance_reward_geometric(self):
    coords, action_coords, highest_coords, lowest_actions = self._pre_reward()
    euclidean_dist = np.linalg.norm(action_coords - coords)

    max_dist = np.linalg.norm(highest_coords - lowest_actions)
    reward = (1 - (euclidean_dist / max_dist)) ** 2

    return abs(reward)


def multiply_reward_geometric(self):
    coords, action_coords, highest_coords, lowest_actions = self._pre_reward()

    single_dists = [
        abs(val1 - val2) for val1, val2 in zip(coords, action_coords, strict=False)
    ]
    max_dists = [
        abs(val1 - val2)
        for val1, val2 in zip(lowest_actions, highest_coords, strict=False)
    ]

    rewards = []
    for dist, max_dist in zip(single_dists, max_dists, strict=False):
        rewards.append(1 - (dist / max_dist))

    return np.prod(rewards)
