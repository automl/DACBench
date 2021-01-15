from stable_baselines import PPO2
from dacbench import benchmarks
from dacbench.logger import Logger
from dacbench.wrappers import PerformanceTrackingWrapper, ObservationWrapper
from pathlib import Path
import argparse


def make_benchmark(config):
    bench = getattr(benchmarks, config["benchmark"])()
    env = bench.get_benchmark(seed=config["seed"])
    if config["benchmark"] in ["SGDBenchmark", "CMAESBenchmark"]:
        env = ObservationWrapper(env)
    wrapped = PerformanceTrackingWrapper(env, logger=config["logger"])
    logger.set_env(wrapped)
    return wrapped


parser = argparse.ArgumentParser(description="Run ray PPO for DACBench")
parser.add_argument("--outdir", type=str, default="output", help="Output directory")
parser.add_argument(
    "--benchmarks", nargs="+", type=str, default=None, help="Benchmarks to run PPO for"
)
parser.add_argument(
    "--timesteps", type=int, default=1000000, help="Number of timesteps to run"
)
parser.add_argument(
    "--seeds",
    nargs="+",
    type=int,
    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    help="Seeds for evaluation",
)
args = parser.parse_args()

for b in args.benchmarks:
    for s in args.seeds:
        logger = Logger(experiment_name=f"PPO_{b}_s{s}", output_path=Path(args.outdir))
        perf_logger = logger.add_module(PerformanceTrackingWrapper)
        logger.set_additional_info(seed=s)
        config = {"seed": s, "logger": perf_logger, "benchmark": b}
        env = make_benchmark(config)
        model = PPO2("MlpPolicy", env)
        model.learn(total_timesteps=args.timesteps)
        logger.close()
