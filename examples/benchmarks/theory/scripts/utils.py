import sys
import yaml
import os
from gym import wrappers

sys.path.append(os.path.dirname(__file__))
# import shutil
# from pprint import pprint

from dacbench.benchmarks import TheoryBenchmark

default_exp_params = {
    "n_cores": 1,
    "n_episodes": 1e6,
    "n_steps": 1e6,
    "eval_interval": 2000,
    "eval_n_episodes": 50,
    "save_agent_at_every_eval": False,
    "seed": 123,
    "use_formula": True
    }

default_bench_params = {
    "name": "Theory",
    "alias": "evenly_spread",
    "discrete_action": True,
    "action_choices": [1,17,33],
    "problem": "LeadingOne",
    "instance_set_path": "lo_rls_50_random.csv",
    "observation_description": "n,f(x)",
    "reward_choice": "imp_minus_evals",
    "seed": 123
    }

default_eval_env_params = {
    "reward_choice": "minus_evals",
    "cutoff": 1e5,
    }


def read_config(config_yml_fn: str = "output/config.yml"):
    with open(config_yml_fn, "r") as f:
        params = yaml.safe_load(f)

    for key in default_exp_params:
        if key not in params["experiment"]:
            params["experiment"][key] = default_exp_params[key]

    for key in default_bench_params:
        if key not in params["bench"]:
            params["bench"][key] = default_bench_params[key]

    train_env_params = eval_env_params = None
    if "train_env" in params:
        train_env_params = params["train_env"]
    if "eval_env" in params:
        eval_env_params = params["eval_env"]
        for key in default_eval_env_params:
            if key not in eval_env_params:
                eval_env_params[key] = default_eval_env__params[key]
    return (
        params["experiment"],
        params["bench"],
        params["agent"],
        train_env_params,
        eval_env_params,
    )


def make_env(bench_params, env_config=None, test_env=False):
    """
    env_config will override bench_params
    """
    bench_class = globals()[bench_params["name"] + "Benchmark"]

    params = bench_params.copy()
    del params["name"]
    if env_config:
        for name, val in env_config.items():
            params[name] = val

    # pprint(params)
    bench = bench_class(config=params)
    env = bench.get_environment(test_env)
    env = wrappers.FlattenObservation(env)
    return env
