import numpy as np
import os
import json
import argparse
from dacbench import benchmarks
from dacbench.wrappers import PerformanceTrackingWrapper
from dacbench.runner import run_benchmark, RandomAgent, StaticAgent, GenericAgent
from dacbench.envs.policies.optimal_sigmoid import get_optimum as optimal_sigmoid
from dacbench.envs.policies.optimal_luby import get_optimum as optimal_luby
from dacbench.envs.policies.optimal_fd import get_optimum as optimal_fd
import itertools

modea_actions = [
    np.arange(2),
    np.arange(2),
    np.arange(2),
    np.arange(2),
    np.arange(2),
    np.arange(2),
    np.arange(2),
    np.arange(2),
    np.arange(2),
    np.arange(3),
    np.arange(3),
]
DISCRETE_ACTIONS = {
    "SigmoidBenchmark": np.arange(int(np.prod((5, 10)))),
    "LubyBenchmark": np.arange(6),
    "FastDownwardBenchmark": [0, 1],
    "CMAESBenchmark": [np.around(a, decimals=1) for a in np.linspace(0.2, 10, num=50)],
    "ModeaBenchmark": list(itertools.product(*modea_actions)),
}


def run_random(results_path, benchmark_name, num_episodes, seeds=np.arange(10)):
    bench = getattr(benchmarks, benchmark_name)()
    for s in seeds:
        env = bench.get_benchmark(seed=s)
        env = PerformanceTrackingWrapper(env)
        agent = RandomAgent(env)
        run_benchmark(env, agent, num_episodes)
        performance = env.get_performance()[0]
        filedir = results_path + "/" + benchmark_name + "/random"
        filename = f"{filedir}/seed_{s}.json"

        if not os.path.exists(results_path):
            os.makedirs(results_path)
        if not os.path.exists(results_path + "/" + benchmark_name):
            os.makedirs(results_path + "/" + benchmark_name)
        if not os.path.exists(filedir):
            os.makedirs(filedir)

        with open(filename, "w+") as fp:
            json.dump(performance, fp)


def run_static(results_path, benchmark_name, action, num_episodes, seeds=np.arange(10)):
    bench = getattr(benchmarks, benchmark_name)()
    for s in seeds:
        env = bench.get_benchmark(seed=s)
        env = PerformanceTrackingWrapper(env)
        agent = StaticAgent(env, action)
        run_benchmark(env, agent, num_episodes)
        performance = env.get_performance()[0]
        filedir = results_path + "/" + benchmark_name + "/static_" + str(action)
        filename = f"{filedir}/seed_{s}.json"

        if not os.path.exists(results_path):
            os.makedirs(results_path)
        if not os.path.exists(results_path + "/" + benchmark_name):
            os.makedirs(results_path + "/" + benchmark_name)
        if not os.path.exists(filedir):
            os.makedirs(filedir)

        with open(filename, "w+") as fp:
            json.dump(performance, fp)


def run_optimal(results_path, benchmark_name, num_episodes, seeds=np.arange(10)):
    bench = getattr(benchmarks, benchmark_name)()
    if benchmark_name == "LubyBenchmark":
        policy = optimal_luby
    elif benchmark_name == "SigmoidBenchmark":
        policy = optimal_sigmoid
    elif benchmark_name == "FastDownwardBenchmark":
        policy = optimal_fd
    else:
        print("No optimal policy found for this benchmark")
        return

    for s in seeds:
        env = bench.get_benchmark(seed=s)
        env = PerformanceTrackingWrapper(env)
        agent = GenericAgent(env, policy)
        run_benchmark(env, agent, num_episodes)
        performance = env.get_performance()[0]
        filedir = results_path + "/" + benchmark_name + "/optimal"
        filename = f"{filedir}/seed_{s}.json"

        if not os.path.exists(results_path):
            os.makedirs(results_path)
        if not os.path.exists(results_path + "/" + benchmark_name):
            os.makedirs(results_path + "/" + benchmark_name)
        if not os.path.exists(filedir):
            os.makedirs(filedir)

        with open(filename, "w+") as fp:
            json.dump(performance, fp)


def main():
    parser = argparse.ArgumentParser(
        description="Run simple baselines for DAC benchmarks"
    )
    parser.add_argument("--outdir", type=str, help="Output directory")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        type=str,
        default=None,
        help="Benchmarks to run baselines for",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate policy on",
    )
    parser.add_argument("--random", action="store_true", help="Run random policy")
    parser.add_argument("--static", action="store_true", help="Run static policy")
    parser.add_argument(
        "--optimal",
        action="store_true",
        help="Run optimal policy. Not available for all benchmarks!",
    )
    parser.add_argument(
        "--dyna_baseline",
        action="store_true",
        help="Run dynamic baseline. Not available for all benchmarks!",
    )
    parser.add_argument(
        "--actions",
        nargs="+",
        type=float,
        default=None,
        help="Action(s) for static policy",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        help="Seeds for evaluation",
    )
    args = parser.parse_args()

    if args.benchmarks is None:
        benchs = benchmarks.__all__
    else:
        benchs = args.benchmarks

    if args.random:
        for b in benchs:
            run_random(args.outdir, b, args.num_episodes, args.seeds)

    if args.static:
        for b in benchs:

            if args.actions is None:
                actions = DISCRETE_ACTIONS[b]
            else:
                actions = args.actions
                if b == "FastDownwardBenchmark":
                    actions = [int(a) for a in actions]
            for a in actions:
                run_static(args.outdir, b, a, args.num_episodes, args.seeds)

    if args.optimal:
        for b in benchs:
            if b not in ["LubyBenchmark", "SigmoidBenchmark", "FastDownwardBenchmark"]:
                print("Optimal policy not available!")
                break

            run_optimal(args.outdir, b, args.num_episodes, args.seeds)


if __name__ == "__main__":
    main()
