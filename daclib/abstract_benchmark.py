class AbstractBenchmark():
    def __init__(self, config_path="some_path"):
        self.config_path = config_path
        self.config = read_config_file(self.config_path)

    def get_config(self):
        return self.config

    def read_config_file(self, path):
        return

    def get_benchmark_env(self):
        #TODO: specify reward_function
        #TODO: specify state space
        #TODO: specify action space
        return
