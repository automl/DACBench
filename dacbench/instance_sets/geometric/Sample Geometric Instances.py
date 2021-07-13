import numpy as np
import os

FILE_PATH = os.path.dirname(__file__)

# Configure amount of different layers
FUNCTION_CONFIG = {
    "sigmoid": 2,
    "linear": 4,
    "polynomial2D": 3,
    "polynomial3D": 2,
    "polynomial7D": 1,
    "exponential": 2,
    "logarithmic": 2,
}

# Each function needs fix number of parameters
FUNCTION_PARAMETER_NUMBERS = {
    "sigmoid": 2,
    "linear": 2,
    "polynomial2D": 3,
    "polynomial3D": 4,
    "polynomial7D": 8,
    "exponential": 1,
    "logarithmic": 1,
}

SAMPLE_SIZE = 100


def save_geometric_instances(filename: str):
    csv_path = os.path.join(FILE_PATH, filename)

    with open(csv_path, "a") as fh:
        id_string = (
            "ID,fcn_name,param1,param2,param3,param4,param5,param6,param7,param8"
        )
        fh.write(id_string)

        for index in range(SAMPLE_SIZE):
            for func_name, count in FUNCTION_CONFIG.items():
                for _ in range(count):
                    instance_string = _create_csv_string(index, func_name)
                    fh.write(instance_string)


def _create_csv_string(index, func_name: str) -> str:
    """
    Create comma separated string with function name and parameter values.
    Set 0 for irrelevant params.

    Parameters
    ----------
    index:
        instance index
    func_name : str
        name of function

    Returns
    -------
    str
        comma separated string
    """
    count = FUNCTION_PARAMETER_NUMBERS[func_name]
    max_count = max(list(FUNCTION_PARAMETER_NUMBERS.values()))

    csv_string = str(index) + "," + func_name

    for i in range(max_count):
        if i < count:
            csv_string += "," + str(np.random.uniform(low=-1.0, high=1.0))
        else:
            csv_string += ",0"

    csv_string += "\n"
    return csv_string


if __name__ == "__main__":
    save_geometric_instances("geometric_train.csv")
    save_geometric_instances("geometric_test.csv")