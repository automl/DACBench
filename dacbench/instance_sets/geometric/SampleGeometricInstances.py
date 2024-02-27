"""Sample geometric instance sets."""
from __future__ import annotations

from pathlib import Path

import numpy as np

FILE_PATH = Path(__file__).parent
rng = np.random.default_rng()

# Configure amount of different layers
FUNCTION_CONFIG = {
    "sigmoid": 1,
    "linear": 1,
    "parabel": 1,
    "cubic": 1,
    "logarithmic": 1,
    "constant": 1,
    "sinus": 1,
}

# Each function needs fix number of parameters
FUNCTION_PARAMETER_NUMBERS = {
    "sigmoid": 2,
    "linear": 2,
    "parabel": 3,
    "cubic": 3,
    "logarithmic": 1,
    "constant": 1,
    "sinus": 1,
}

SAMPLE_SIZE = 100


def save_geometric_instances(
    filename: str, config: dict = FUNCTION_CONFIG, path: str = ""
):
    """First delete old isntance_set.

    Create new instances based on config.

    Parameters
    ----------
    filename : str
        name of instance set
    config : Dict, optional
        config that has info about which functions will get selected,
        by default FUNCTION_CONFIG
    path : std
        path to save to

    """
    csv_path = Path(path) / filename if path else Path(FILE_PATH) / filename

    if Path.exists(csv_path):
        Path.unlink(csv_path)

    with open(csv_path, "a") as fh:
        id_string = "ID,fcn_name"
        for index in range(1, max(list(FUNCTION_PARAMETER_NUMBERS.values())) + 1):
            id_string += f",param{index}"
        id_string += "\n"

        fh.write(id_string)

        for index in range(SAMPLE_SIZE):
            for func_name, count in config.items():
                for _ in range(count):
                    instance_string = _create_csv_string(index, func_name)
                    fh.write(instance_string)


def _create_csv_string(index, func_name: str) -> str:
    """Create comma separated string with function name and parameter values.
    Set 0 for irrelevant params.

    Parameters
    ----------
    index:
        instance index
    func_name : str
        name of function

    Returns:
    -------
    str
        comma separated string

    """
    count = FUNCTION_PARAMETER_NUMBERS[func_name]
    max_count = max(list(FUNCTION_PARAMETER_NUMBERS.values()))

    csv_string = str(index) + "," + func_name

    if func_name == "sigmoid":
        value_generator = sample_sigmoid_value()
    elif func_name in ("cubic", "parabel"):
        value_generator = sample_parabel_cubic_value()

    for i in range(max_count):
        if i < count:
            if func_name == "sinus":
                value = np.round(sample_sinus_value(), 1)
            elif func_name == "sigmoid":
                value = np.round(next(value_generator), 1)
            elif func_name in ("cubic", "parabel"):
                value = next(value_generator)
            else:
                value = np.round(rng.uniform(low=-10.0, high=10.0), 1)

            csv_string += "," + str(value)
        else:
            csv_string += ",0"

    csv_string += "\n"
    return csv_string


def sample_sinus_value():
    """Get values for sinus."""
    return np.round(rng.uniform(low=0.5, high=2.0), 1)


def sample_sigmoid_value():
    """Get values for sigmoid."""
    scale = np.round(rng.uniform(low=0.1, high=4.0), 1)
    yield scale
    infliction = np.round(rng.uniform(low=0, high=10), 1)
    yield infliction


def sample_parabel_cubic_value():
    """Get values for cubic."""
    sig = [-1, 1]
    yield rng.choice(sig)

    x_int = list(range(3, 8))
    yield rng.choice(x_int)

    y_int = [-50, -20, -10, -5, -1, 0, 1, 5, 10, 20, 50]
    yield rng.choice(y_int)


if __name__ == "__main__":
    # save_geometric_instances("geometric_unit_test.csv", FUNCTION_CONFIG)
    save_geometric_instances("geometric_test.csv", FUNCTION_CONFIG)
