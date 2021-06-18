from stable_baselines import PPO2
from stable_baselines.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    CallbackList,
)
from dacbench import benchmarks
from dacbench.logger import Logger
from dacbench.wrappers import PerformanceTrackingWrapper, ObservationWrapper
from pathlib import Path
import argparse


class LoggerCallback(BaseCallback):
    def __init__(self, logger, verbose=0):
        super(LoggerCallback, self).__init__(verbose)
        self.env_logger = logger

    def _on_step(self):
        self.env_logger.next_step()
        return True

    def _on_rollout_end(self):
        self.env_logger.next_episode()


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
parser.add_argument("--fd_port", type=int, default=55555)
args = parser.parse_args()

for b in args.benchmarks:
    for s in args.seeds:
        logger = Logger(experiment_name=f"PPO_{b}_s{s}", output_path=Path(args.outdir))
        perf_logger = logger.add_module(PerformanceTrackingWrapper)
        config = {"seed": s, "logger": perf_logger, "benchmark": b}
        if b == "FastDownwardBenchmark":
            config["port"] = args.fd_port
        env = make_benchmark(config)
        model = PPO2("MlpPolicy", env)
        logging = LoggerCallback(logger)

        checkpoint = CheckpointCallback(
            save_freq=1000,
            save_path=f"{args.outdir}/PPO_{b}_s{s}/models",
            name_prefix="model",
        )
        callback = CallbackList([logging, checkpoint])
        model.learn(total_timesteps=args.timesteps, callback=callback)
        logger.close()
