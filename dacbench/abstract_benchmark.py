import json
import numpy as np


class AbstractBenchmark:
    """
    Abstract template for benchmark classes
    """

    def __init__(self, config_path=None):
        """
        Initialize benchmark class

        Parameters
        -------
        config_path : str
            Path to load configuration from (if read from file)
        """
        if config_path:
            self.config_path = config_path
            self.read_config_file(self.config_path)
        else:
            self.config = None

    def get_config(self):
        """
        Return current configuration

        Returns
        -------
        dict
            Current config
        """
        return self.config

    def save_config(self, path):
        """
        Save configuration to .json

        Parameters
        ----------
        path : str
            File to save config to
        """
        conf = self.config.copy()
        if "observation_space_type" in self.config:
            conf["observation_space_type"] = f"{self.config['observation_space_type']}"
        for k in self.config.keys():
            if isinstance(self.config[k], np.ndarray) or isinstance(
                self.config[k], list
            ):
                if type(self.config[k][0]) == np.ndarray:
                    conf[k] = list(map(list, conf[k]))
                    for i in range(len(conf[k])):
                        if (
                            not type(conf[k][i][0]) == float
                            and np.inf not in conf[k][i]
                            and -np.inf not in conf[k][i]
                        ):
                            conf[k][i] = list(map(int, conf[k][i]))
        with open(path, "w") as fp:
            json.dump(conf, fp)

    def read_config_file(self, path):
        """
        Read configuration from file

        Parameters
        ----------
        path : str
            Path to config file
        """
        with open(path, "r") as fp:
            self.config = objdict(json.load(fp))
        if "observation_space_type" in self.config:
            # Typwa have to be numpy dtype (for gym spaces)s
            if type(self.config["observation_space_type"]) == str:
                typestring = self.config["observation_space_type"].split(" ")[1][:-2]
                typestring = typestring.split(".")[1]
                self.config["observation_space_type"] = getattr(np, typestring)
        for k in self.config.keys():
            if type(self.config[k]) == list:
                if type(self.config[k][0]) == list:
                    map(np.array, self.config[k])
                self.config[k] = np.array(self.config[k])

    def get_environment(self):
        """
        Make benchmark environment

        Returns
        -------
        env : gym.Env
            Benchmark environment
        """
        raise NotImplementedError

    def set_seed(self, seed):
        """
        Set environment seed

        Parameters
        ----------
        seed : int
            New seed
        """
        self.config["seed"] = seed

    def set_action_space(self, kind, args):
        """
        Change action space

        Parameters
        ----------
        kind : str
            Name of action space class
        args: list
            List of arguments to pass to action space class
        """
        self.config["action_space"] = kind
        self.config["action_space_args"] = args

    def set_observation_space(self, kind, args, data_type):
        """
        Change observation_space

        Parameters
        ----------
        config : str
            Name of observation space class
        args : list
            List of arguments to pass to observation space class
        data_type : type
            Data type of observation space
        """
        self.config["observation_space"] = kind
        self.config["observation_space_args"] = args
        self.config["observation_space_type"] = data_type


# This code is taken from https://goodcode.io/articles/python-dict-object/
class objdict(dict):
    """
    Modified dict to make config changes more flexible
    """

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)
