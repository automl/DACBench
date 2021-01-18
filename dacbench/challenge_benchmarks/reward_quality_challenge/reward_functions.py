import numpy as np


def easy_sigmoid(self):
    sigmoids = [
        np.abs(self._sig(self.c_step, slope, shift))
        for slope, shift in zip(self.shifts, self.slopes)
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
        r += 10 ** i * action_diffs[i]
    r = max(self.reward_range[0], min(self.reward_range[1], r))
    return r


def almost_easy_sigmoid(self):
    r = [
        1 - np.abs(self._sig(self.c_step, slope, shift) - (act / (max_act - 1)))
        for slope, shift, act, max_act in zip(
            self.slopes, self.shifts, self.action, self.action_vals
        )
    ]
    r = sum(r)
    r = max(self.reward_range[0], min(self.reward_range[1], r))
    return r


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
