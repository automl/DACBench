import json

class AbstractBenchmark():
    def __init__(self, config_path):
        if config_path:
            self.config_path = config_path
            self.config = read_config_file(self.config_path)

        if not self.config:
            self.config = {"seed": 0}

    def get_config(self):
        return self.config

    def save_config(self, path):
        with open(path, 'w') as fp:
            json.dump(self.config, fp)

    def read_config_file(self, path):
        with open(path, 'r') as fp:
            self.config = json.load(fp)

    def get_benchmark_env(self):
        raise NotImplementedError

    def set_seed(self, seed):
        self.config["seed"] = seed
