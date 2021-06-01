# from dacbench.envs import OneLLEnv
from dacbench.benchmarks import OneLLBenchmark
from dacbench.envs.onell_env import OneMax  # , LeadingOne
from dacbench.envs.policies.bestknown_onell import get_dyn_theory, get_dyn_onefifth

import numpy as np
import os

import unittest


def get_default_rng(seed):
    return np.random.default_rng(seed)


def onell_dynamic_theory(
    n,
    problem=OneMax,
    seed=None,
    max_evals=99999999,
    count_different_inds_only=True,
    include_xprime_crossover=True,
):
    """
    (1+LL)-GA, dynamic version with theoretical results
    lbd = sqrt(n*(n-f(x))), p = lbd/n, c=1/lbd
    """
    rng = get_default_rng(seed)

    x = problem(n, rng=rng)
    f_x = x.fitness

    total_evals = 1  # total number of solution evaluations

    # mtimes, ctimes = [], []
    steps = 1
    # old_f_x = f_x
    while not x.is_optimal():
        # mutation phase
        lbd = np.sqrt(n / (n - f_x))
        lbd = int(lbd)
        p = lbd / n
        xprime, f_xprime, ne1 = x.mutate(p, int(lbd), rng)

        # crossover phase
        c = 1 / lbd
        y, f_y, ne2 = x.crossover(
            xprime,
            c,
            int(lbd),
            include_xprime_crossover,
            count_different_inds_only,
            rng,
        )

        # selection phase
        #    old_f_x = f_x
        if f_x <= f_y:
            x = y
            f_x = f_y

        total_evals = total_evals + ne1 + ne2

        # print("%d: evals=%d; x=%d; xprime=%d; y=%d; obj=%d; p=%.2f; c=%.2f; lbd=%.2f" % (steps, total_evals, old_f_x,xprime.fitness, y.fitness, x.fitness, p, c, lbd))

        steps += 1

        if total_evals >= max_evals:
            break

    # print(total_evals)

    return x, f_x, total_evals


def onell_dynamic_5params(
    n,
    problem=OneMax,
    seed=None,
    alpha=0.45,
    beta=1.6,
    gamma=1,
    A=1.16,
    b=0.7,
    max_evals=99999999,
    count_different_inds_only=True,
    include_xprime_crossover=True,
):
    """
    (1+LL)-GA, dynamic version with 5 hyper-parameters as in https://arxiv.org/pdf/1904.04608.pdf
    The default hyper-parameter setting here is the best one found in that paper

    Arguments:

    Returns:

    """
    assert A > 1 and b < 1

    rng = get_default_rng(seed)

    x = problem(n, rng=rng)
    f_x = x.fitness

    lbd = 1
    min_prob = 1 / n
    max_prob = 0.99

    total_evals = 1  # total number of solution evaluations

    # mtimes, ctimes = [], []
    steps = 0
    while not x.is_optimal():
        # mutation phase
        p = np.clip(alpha * lbd / n, min_prob, max_prob)
        xprime, f_xprime, ne1 = x.mutate(p, int(lbd), rng)

        # print(np.clip(lbd,1,n))

        # crossover phase
        c = np.clip(gamma / lbd, min_prob, max_prob)
        n_childs = round(lbd * beta)
        y, f_y, ne2 = x.crossover(
            xprime,
            c,
            n_childs,
            include_xprime_crossover,
            count_different_inds_only,
            rng,
        )

        # update parameters
        if x.fitness < y.fitness:
            lbd = max(b * lbd, 1)
        else:
            lbd = min(A * lbd, n - 1)

        # selection phase
        # old_f_x = f_x
        if f_x <= f_y:
            x = y
            f_x = f_y

        total_evals = total_evals + ne1 + ne2

        steps += 1
        # print("%d: evals=%d; x=%d; xprime=%d; y=%d; obj=%d; p=%.2f; c=%.2f; lbd=%.2f" % (steps, total_evals, old_f_x,xprime.fitness, y.fitness, x.fitness, p, c, lbd))

        if total_evals >= max_evals:
            break

    # print(total_evals)

    return x, f_x, total_evals


class TestOneLLBaselines(unittest.TestCase):
    def test_dyn_theory(self):
        config_path = (
            os.path.dirname(os.path.abspath(__file__))
            + "/../../dacbench/additional_configs/onell/lbd_theory.json"
        )
        bench = OneLLBenchmark(config_path)
        bench.config.instance_set_path = "../instance_sets/onell/onemax_2000.csv"
        bench.read_instance_set()
        env = bench.get_environment()
        n = 2000

        n_runs = 10
        sum_evals_RL = 0
        sum_evals_non_RL = 0
        for i in range(n_runs):
            # run using RL setting (OneLLEnv)
            state = env.reset()
            done = False
            while not done:
                action = get_dyn_theory(env, state)
                state, reward, done, _ = env.step(action)
            sum_evals_RL += env.total_evals

            # run using non RL setting
            n_evals = onell_dynamic_theory(n=n)[2]
            sum_evals_non_RL += n_evals

            # print("%d, %d" % (env.total_evals, n_evals))

        # print("avg evals by OneLLEnv: " + str(sum_evals_RL / n_runs))
        # print("avg evals by non-RL setting: " + str(sum_evals_non_RL / n_runs))

        # make sure the two implementations are not too different from each other
        ratio = (sum_evals_RL / n_runs) / (sum_evals_non_RL / n_runs)
        assert ratio >= 0.9 and ratio <= 1.1

    def test_dyn_onefifth(self):
        config_path = (
            os.path.dirname(os.path.abspath(__file__))
            + "/../../dacbench/additional_configs/onell/lbd_onefifth.json"
        )
        bench = OneLLBenchmark(config_path)
        bench.config.instance_set_path = "../instance_sets/onell/onemax_2000.csv"
        bench.read_instance_set()
        env = bench.get_environment()
        n = 2000

        n_runs = 5
        sum_evals_RL = 0
        sum_evals_non_RL = 0
        for i in range(n_runs):
            # run using RL setting (OneLLEnv)
            state = env.reset()
            done = False
            while not done:
                action = get_dyn_onefifth(env, state)
                state, reward, done, _ = env.step(action)
            sum_evals_RL += env.total_evals

            # run using non RL setting
            n_evals = onell_dynamic_5params(
                n=n, alpha=1, beta=1, gamma=1, A=1.107, b=0.67
            )[2]
            sum_evals_non_RL += n_evals

            # print("%d, %d" % (env.total_evals, n_evals))

        # print("avg evals by OneLLEnv: " + str(sum_evals_RL / n_runs))
        # print("avg evals by non-RL setting: " + str(sum_evals_non_RL / n_runs))

        # make sure the two implementations are not too different from each other
        ratio = (sum_evals_RL / n_runs) / (sum_evals_non_RL / n_runs)
        assert ratio >= 0.9 and ratio <= 1.1


# TestOneLLBaselines().test_dyn_theory()
# TestOneLLBaselines().test_dyn_onefifth()
