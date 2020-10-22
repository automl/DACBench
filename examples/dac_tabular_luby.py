"""
Code adapted from
"Dynamic Algorithm Configuration:Foundation of a New Meta-Algorithmic Framework"
by A. Biedenkapp and H. F. Bozkurt and T. Eimer and F. Hutter and M. Lindauer.
Original environment authors: André Biedenkapp, H. Furkan Bozkurt
"""

import argparse
import logging
import numpy as np
import datetime
import pickle
import os
from example_utils import q_learning, zeroOne
from dacbench.benchmarks import LubyBenchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Tabular Q-learning example")
    parser.add_argument(
        "-n",
        "--n-eps",
        dest="neps",
        default=100,
        help="Number of episodes to roll out.",
        type=int,
    )
    parser.add_argument(
        "--epsilon_decay",
        choices=["linear", "log", "const"],
        default="const",
        help="How to decay epsilon from the given starting point to 0 or constant",
    )
    parser.add_argument(
        "--decay_starts",
        type=int,
        default=0,
        help="How long to keep epsilon constant before decaying. "
        "Only active if epsilon_decay log or linear",
    )
    parser.add_argument(
        "-r",
        "--repetitions",
        default=1,
        help="Number of repeated learning runs.",
        type=int,
    )
    parser.add_argument(
        "-d",
        "--discount-factor",
        dest="df",
        default=0.99,
        help="Discount factor",
        type=zeroOne,
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        dest="lr",
        default=0.125,
        help="Learning rate",
        type=float,
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        default=0.1,
        help="Epsilon for the epsilon-greedy method to follow",
        type=zeroOne,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Use debug output")
    parser.add_argument(
        "-o",
        "--out-dir",
        dest="out_dir",
        default=".",
        help="Dir to store result files in",
    )
    parser.add_argument(
        "--instance_feature_file",
        dest="inst_feats",
        default="../instance_sets/luby_train.csv",
        help="Instance feature file to use for sigmoid environment",
        type=str,
    )
    parser.add_argument("-s", "--seed", default=0, type=int)
    parser.add_argument("--cutoff", default=None, type=int, help="Env max steps")
    parser.add_argument(
        "--min_steps",
        default=None,
        type=int,
        help="Env min steps. Only active for the lubyt* environments",
    )
    parser.add_argument(
        "--test_insts",
        default=None,
        help="Test instances to use with q-learning for evaluation purposes",
        type=str,
    )
    parser.add_argument(
        "--reward_variance",
        default=1.5,
        type=float,
        help="Variance of noisy reward signal for lubyt*",
    )

    args = parser.parse_args()
    benchmark = LubyBenchmark()
    if args.cutoff:
        benchmark.config.cutoff = args.cutoff
    if args.min_steps:
        benchmark.config.min_steps = args.min_steps
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    np.random.seed(args.seed)
    stats, tstats = [], []
    rewards = []
    lens = []
    expects = []
    q_func = None
    if not os.path.exists(os.path.realpath(args.out_dir)):
        os.mkdir(args.out_dir)

    ds = datetime.date.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S_%f")
    random_agent = False
    if args.epsilon == 1.0 and args.epsilon_decay == "const":
        folder = "random_" + ds
        random_agent = True
    else:
        folder = "tabular_" + ds
    folder = os.path.join(os.path.realpath(args.out_dir), folder)
    if not os.path.exists(folder):
        os.mkdir(folder)
        os.chdir(folder)
    for r in range(args.repetitions):
        logging.info("Round %d", r)
        benchmark.config.seed = args.seed
        benchmark.config.instance_set_path = args.inst_feats
        env = benchmark.get_benchmark_env()

        if args.test_insts:
            benchmark.config.seed = 2 ** 32 - 1 - args.seed
            benchmark.config.instance_set_path = args.test_insts
            test_env = benchmark.get_benchmark_env()
        else:
            test_env = None

        eval_for = 100
        float_state = False
        q_func, test_train_stats = q_learning(
            env,
            args.neps,
            discount_factor=args.df,
            alpha=args.lr,
            epsilon=args.epsilon,
            verbose=args.verbose,
            track_test_stats=True if test_env else False,
            float_state=float_state,
            epsilon_decay=args.epsilon_decay,
            decay_starts=args.decay_starts,
            number_of_evaluations=eval_for,
            test_environment=test_env,
        )
        fn = "%04d_%s-greedy-results-luby-%d_eps-%d_reps-seed_%d.pkl" % (
            r,
            str(args.epsilon),
            args.neps,
            args.repetitions,
            args.seed,
        )
        with open(fn.replace("results", "q_func"), "wb") as fh:
            pickle.dump(dict(q_func), fh)
        with open(fn.replace("results", "stats"), "wb") as fh:
            pickle.dump(test_train_stats[1].episode_rewards, fh)
        with open(fn.replace("results", "lens"), "wb") as fh:
            pickle.dump(test_train_stats[1].episode_lengths, fh)
        if args.test_insts:
            with open(fn.replace("results", "test_stats"), "wb") as fh:
                pickle.dump(test_train_stats[0].episode_rewards, fh)
            with open(fn.replace("results", "test_lens"), "wb") as fh:
                pickle.dump(test_train_stats[0].episode_lengths, fh)
        if r == args.repetitions - 1:
            logging.info("Example Q-function:")
            for i in range(10):
                print("#" * 120)
                s = env.reset()
                done = False
                while not done:
                    q_vals = q_func[s]
                    action = np.argmax(q_vals)
                    print(s, q_vals, action)
                    s, _, done, _ = env.step(action)
