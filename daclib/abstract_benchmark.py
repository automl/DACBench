import json


class AbstractBenchmark:
    """
    Abstract template for benchmark classes
    """

    def __init__(self, config_path=None):
        if config_path:
            self.config_path = config_path
            self.config = self.read_config_file(self.config_path)
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
        with open(path, "w") as fp:
            json.dump(self.config, fp)

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

    def get_benchmark_env(self):
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


# TODO: source!
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
