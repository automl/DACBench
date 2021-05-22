"""
All configs for the (1+(lambda, lambda)) Genetic Algorithms
To generate the .json config files in this folder, run: export_all_configs_to_json()
"""

from copy import copy
from dacbench.abstract_benchmark import objdict, AbstractBenchmark
import numpy as np


INFO = {
    "identifier": "onell",
    "name": "OneLL-GA benchmark",
    "reward": "",
    "state_description": [""],
}

"""
Setting 1a: 
  Name: lbd_theory
  Description: 
      The simplest setting where we only tune lbd and assume that p=lambda/n and c=1/lambda
      The optimal policy for OneMax is lambda = sqrt(n / (n-f(x))), where:
          n is problem size
          f(x) is current objective value
  State space: n (int), f(x) (int)
  Action space: lbd (int)
"""
onell_lbd_theory = objdict(
    {
        "name": "lbd_theory",
        "action_space_class": "Box",
        "action_space_args": [np.array([1]), np.array([np.inf])],
        "action_description": "lbd",
        "observation_space_class": "Box",
        "observation_space_type": np.int32,
        "observation_space_args": [np.array([1, 0]), np.array([np.inf, np.inf])],
        "observation_description": "n, f(x)",
        "reward_range": [
            -np.inf,
            np.inf,
        ],  # the true reward range is instance dependent
        "cutoff": 1e9,  # we don't really use this,
        # the real cutoff is in instance_set_path and is instance dependent
        "include_xprime": True,  # if True, xprime is included in the selection after crossover phase
        "count_different_inds_only": True,  # if True, only count an evaluation of a child if it is different from both of its parents
        "seed": 0,
        "problem": "OneMax",
        "instance_set_path": "../instance_sets/onell/onemax_2000.csv",
        "benchmark_info": INFO,
    }
)

# Setting 1b:
#   Name: lbd_onefifth
#   Description:
#       Same as setting 1a but with slightly different state-space (n, delta f(x), lambda_{t-1})
#       In this setting, the agent can learn the best policy for OneMax using the 1/5th rule:
#           if delta f(x) = f(x_t) - f(x_{t-1}) <= 0: lambda_t = min{(3/2)^1/4 * lambda_{t-1}, n-1}
#           otherwise: lambda_t = max{2/3 * lambda_{t-1}, 1}
onell_lbd_onefifth = copy(onell_lbd_theory)
onell_lbd_onefifth["name"] = "lbd_onefifth"
onell_lbd_onefifth["observation_space_type"] = np.float32
onell_lbd_onefifth["observation_space_args"] = [
    np.array([1, 0, 1]),
    np.array([np.inf, np.inf, np.inf]),
]
onell_lbd_onefifth["observation_description"] = "n, delta f(x), lbd_{t-1}"


# Setting 2:
#   Name: lbd_p_c
#   Description: a more sophisticated setting where we tune lambda, p and c together
#   State space: n (int), f(x) (int), delta f(x) (int), lambda_{t-1}, p_{t-1}, c_{t-1}
#   action space: lambda (int), p (float), c (float)
onell_lbd_p_c = copy(onell_lbd_theory)
onell_lbd_p_c["name"] = "lbd_p_c"
onell_lbd_p_c["observation_space_type"] = np.float32
onell_lbd_p_c["observation_space_args"] = [
    np.array([1, 0, 0, 1, 0, 0]),
    np.array([np.inf, np.inf, np.inf, np.inf, 1, 1]),
]
onell_lbd_p_c["action_space_args"] = [np.array([1, 0, 0]), np.array([np.inf, 1, 1])]
onell_lbd_p_c["action_description"] = "lbd, p, c"
onell_lbd_p_c[
    "observation_description"
] = "n, f(x), delta f(x), lbd_{t-1}, p_{t-1}, c_{t-1}"

# Setting 3:
#   Name: lbd1_lbd2_p_c
#   Description: a setting where we tune lambda1 (#mutated off-springs), lambda2 (#crossovered off-springs), p and c together
#   State space: n (int), f(x) (int), delta f(x) (int), lambda1_{t-1}, lambda2_{t-1}, p_{t-1}, c_{t-1}
#   action space: lambda1 (int), lambda2 (int), p (float), c (float)
onell_lbd1_lbd2_p_c = copy(onell_lbd_theory)
onell_lbd1_lbd2_p_c["name"] = "lbd1_lbd2_p_c"
onell_lbd1_lbd2_p_c["observation_space_type"] = np.float32
onell_lbd1_lbd2_p_c["observation_space_args"] = [
    np.array([1, 0, 0, 1, 1, 0, 0]),
    np.array([np.inf, np.inf, np.inf, np.inf, np.inf, 1, 1]),
]
onell_lbd1_lbd2_p_c["action_space_args"] = [
    np.array([1, 1, 0, 0]),
    np.array([np.inf, np.inf, 1, 1]),
]
onell_lbd1_lbd2_p_c["action_description"] = "lbd1, lbd2, p, c"
onell_lbd1_lbd2_p_c[
    "observation_description"
] = "n, f(x), delta f(x), lbd1_{t-1}, lbd2_{t-1}, p_{t-1}, c_{t-1}"


def config_to_json(config_name, json_file=None):
    """
    Write a config to json file

    Parameters
    ---------
    config_name: str
        accept values: one of the configs defined above
    json_file: str
        output .json file name. If None, will be set as <config_name>.json
    """
    if json_file is None:
        json_file = config_name + ".json"
    bench = AbstractBenchmark()
    bench.config = globals()["onell_" + config_name]
    bench.save_config(json_file)


def export_all_configs_to_json(output_dir="./"):
    """
    Export all configs above to json files
    """
    for config_name in ["lbd_theory", "lbd_onefifth", "lbd_p_c", "lbd1_lbd2_p_c"]:
        config_to_json(config_name, json_file=output_dir + "/" + config_name + ".json")


# export_all_configs_to_json()
