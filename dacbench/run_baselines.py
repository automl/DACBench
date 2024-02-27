"""Run policies for baselines."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from dacbench import benchmarks
from dacbench.agents import DynamicRandomAgent, GenericAgent, StaticAgent
from dacbench.envs.policies import NON_OPTIMAL_POLICIES, OPTIMAL_POLICIES
from dacbench.logger import Logger
from dacbench.runner import run_benchmark
from dacbench.wrappers import PerformanceTrackingWrapper


def run_random(results_path, benchmark_name, num_episodes, seeds, fixed):
    """Run random policy.

    Parameters
    ----------
    results_path : str
        Path to where results should be saved
    benchmark_name : str
        Name of the benchmark to run
    num_episodes : int
        Number of episodes to run for each benchmark
    seeds : list[int]
        List of seeds to runs all benchmarks for.
        If None (default) seeds [1, ..., 10] are used.
    fixed : int
        Number of fixed steps per action

    """
    bench = getattr(benchmarks, benchmark_name)()
    for s in seeds:
        experiment_name = f"random_fixed{fixed}_{s}" if fixed > 1 else f"random_{s}"
        logger = Logger(
            experiment_name=experiment_name, output_path=results_path / benchmark_name
        )
        env = bench.get_benchmark(seed=s)
        env = PerformanceTrackingWrapper(
            env, logger=logger.add_module(PerformanceTrackingWrapper)
        )
        agent = DynamicRandomAgent(env, fixed)

        logger.add_agent(agent)
        logger.add_benchmark(bench)
        logger.set_env(env)

        run_benchmark(env, agent, num_episodes, logger)

        logger.close()


def run_static(results_path, benchmark_name, action, num_episodes, seeds=None):
    """Run static policy.

    Parameters
    ----------
    results_path : str
        Path to where results should be saved
    benchmark_name : str
        Name of the benchmark to run
    action : int | float
        The action to run
    num_episodes : int
        Number of episodes to run for each benchmark
    seeds : list[int]
        List of seeds to runs all benchmarks for.
        If None (default) seeds [1, ..., 10] are used.

    """
    if seeds is None:
        seeds = np.arange(10)
    bench = getattr(benchmarks, benchmark_name)()
    for s in seeds:
        logger = Logger(
            experiment_name=f"static_{action}_{s}",
            output_path=results_path / benchmark_name,
        )
        env = bench.get_benchmark(seed=s)
        env = PerformanceTrackingWrapper(
            env, logger=logger.add_module(PerformanceTrackingWrapper)
        )
        agent = StaticAgent(env, action)

        logger.add_agent(agent)
        logger.add_benchmark(bench)
        logger.set_env(env)
        logger.set_additional_info(action=action)

        run_benchmark(env, agent, num_episodes, logger)

        logger.close()


def run_optimal(results_path, benchmark_name, num_episodes, seeds):
    """Run optimal policy.

    Parameters
    ----------
    results_path : str
        Path to where results should be saved
    benchmark_name : str
        Name of the benchmark to run
    num_episodes : int
        Number of episodes to run for each benchmark
    seeds : list[int]
        List of seeds to runs all benchmarks for.
        If None (default) seeds [1, ..., 10] are used.

    """
    if benchmark_name not in OPTIMAL_POLICIES:
        print("No optimal policy found for this benchmark")
        return
    policy = OPTIMAL_POLICIES[benchmark_name]
    run_policy(results_path, benchmark_name, num_episodes, policy, seeds)


def run_dynamic_policy(results_path, benchmark_name, num_episodes, seeds=None):
    """Run dynamic baseline policy.

    Parameters
    ----------
    results_path : str
        Path to where results should be saved
    benchmark_name : str
        Name of the benchmark to run
    num_episodes : int
        Number of episodes to run for each benchmark
    seeds : list[int]
        List of seeds to runs all benchmarks for.
        If None (default) seeds [1, ..., 10] are used.

    """
    if seeds is None:
        seeds = np.arange(10)
    if benchmark_name not in NON_OPTIMAL_POLICIES:
        print("No dynamic policy found for this benchmark")
    policy = NON_OPTIMAL_POLICIES[benchmark_name]
    run_policy(results_path, benchmark_name, num_episodes, policy, seeds)


def run_policy(results_path, benchmark_name, num_episodes, policy, seeds=None):
    """Run generic policy.

    Parameters
    ----------
    results_path : str
        Path to where results should be saved
    benchmark_name : str
        Name of the benchmark to run
    num_episodes : int
        Number of episodes to run for each benchmark
    policy : AbstractDACBenchAgent
        The policy to run
    seeds : list[int]
        List of seeds to runs all benchmarks for.
        If None (default) seeds [1, ..., 10] are used.

    """
    if seeds is None:
        seeds = np.arange(10)
    bench = getattr(benchmarks, benchmark_name)()

    for s in seeds:
        if benchmark_name == "CMAESBenchmark":
            experiment_name = f"csa_{s}"
        else:
            experiment_name = f"optimal_{s}"
        logger = Logger(
            experiment_name=experiment_name,
            output_path=Path(results_path) / benchmark_name,
        )

        env = bench.get_benchmark(seed=s)
        env = PerformanceTrackingWrapper(
            env, logger=logger.add_module(PerformanceTrackingWrapper)
        )

        try:
            agent = policy(env)
        except:  # noqa: E722
            agent = GenericAgent(env, policy)

        logger.add_agent(agent)
        logger.add_benchmark(bench)
        logger.set_env(env)

        run_benchmark(env, agent, num_episodes, logger)

        logger.close()


def main(args):
    """Main evaluation loop."""
    parser = argparse.ArgumentParser(
        description="Run simple baselines for DAC benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--outdir", type=str, default="output", help="Output directory")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        type=str,
        choices=benchmarks.__all__,
        default=None,
        help="Benchmarks to run baselines for, if not provides all benchmarks are run.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate policy on",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help=(
            "Run random policy. Use '--fixed_random' to fix the "
            "random action for a number of steps"
        ),
    )
    parser.add_argument("--static", action="store_true", help="Run static policy")

    parser.add_argument(
        "--optimal",
        action="store_true",
        help=(
            "Run optimal policy. "
            "Only available for {', '.join(OPTIMAL_POLICIES.keys())}"
        ),
    )
    parser.add_argument(
        "--dyna_baseline",
        action="store_true",
        help=(
            "Run dynamic baseline. "
            "Only available for {', '.join(NON_OPTIMAL_POLICIES.keys())}"
        ),
    )
    parser.add_argument(
        "--actions",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Action(s) for static policy. "
            "Make sure, that the actions correspond to the benchmarks."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        help="Seeds for evaluation",
    )
    parser.add_argument(
        "--fixed_random",
        type=int,
        default=0,
        help="Fixes random actions for n steps",
    )
    args = parser.parse_args(args)

    benchs = benchmarks.__all__ if args.benchmarks is None else args.benchmarks

    args.outdir = Path(args.outdir)

    if args.random:
        for b in benchs:
            run_random(args.outdir, b, args.num_episodes, args.seeds, args.fixed_random)

    if args.static:
        for b in benchs:
            if args.actions is None:
                raise ValueError("Missing actions argument for static policy.")
            actions = args.actions
            if b == "FastDownwardBenchmark":
                actions = [int(a) for a in actions]
            for a in actions:
                run_static(args.outdir, b, a, args.num_episodes, args.seeds)

    if args.optimal:
        for b in benchs:
            if b not in OPTIMAL_POLICIES:
                print("Option not available!")
                break

            run_optimal(args.outdir, b, args.num_episodes, args.seeds)

    if args.dyna_baseline:
        for b in benchs:
            if b not in NON_OPTIMAL_POLICIES:
                print("Option not available!")
                break

            run_dynamic_policy(args.outdir, b, args.num_episodes, args.seeds)


if __name__ == "__main__":
    main(sys.argv[1:])
