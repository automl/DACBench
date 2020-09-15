class AbstractBenchmark():
    def __init__(self, config_path="some_path"):
        self.config_path = config_path
        return

    def get_config(self):
        load_config_file(self.config_path)
        return

    def load_config_file(path):
        return

    def get_benchmark_env(self):
        return
