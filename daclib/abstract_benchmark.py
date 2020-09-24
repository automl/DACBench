import json

class AbstractBenchmark():
    def __init__(self, config_path=None):
        if config_path:
            self.config_path = config_path
            self.config = read_config_file(self.config_path)
        else:
            self.config = None

        if not self.config:
            self.config = objdict({"seed": 0})

    def get_config(self):
        return self.config

    def save_config(self, path):
        with open(path, 'w') as fp:
            json.dump(self.config, fp)

    def read_config_file(self, path):
        with open(path, 'r') as fp:
            self.config = objdict(json.load(fp))

    def get_benchmark_env(self):
        raise NotImplementedError

    def set_seed(self, seed):
        self.config["seed"] = seed

    def set_action_space(self, kind, args):
        self.config["action_space"] = kind
        self.config["action_space_args"] = args

    def set_observation_space(self, kind, args, data_type):
        self.config["observation_space"] = kind
        self.config["observation_space_args"] = args
        self.config["observation_space_type"] = data_type

class objdict(dict):
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
