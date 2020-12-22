import json
from collections import defaultdict
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Union, Dict, Any, Tuple, List


class Logger:
    valid_types = {
        "recursive": [dict, list, tuple],
        "primitive": [str, int, float, bool],
    }

    def __pretty_valid_types(self) -> str:
        valid_types = chain(
            self.valid_types["recursive"], self.valid_types["primitive"]
        )
        return ", ".join(map(lambda type: type.__name__, valid_types))

    def __init__(
        self,
        output_path: Path,
        experiment_name: str,
        module: str = None,
        write_every_n_steps: int = 1,
    ) -> None:
        """
        All results are placed under 'output_path / experiment_name'
        :param output_path: the path where logged information should be stored
        :param experiment_name: name of the experiment.
        :param the module (mostly name of the wrapper), each wrapper get's its own file
        """
        # todo write buffer every frequency
        # todo add multi module support
        # todo add registration
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.log_dir = self.__init_logging_dir(self.output_path / self.experiment_name)
        self.log_file = open(self.log_dir / "{module}.jsonl", "a+")

        self.step = 0
        self.episode = 0
        self.buffer = []
        self.current_step = self.__init_dict()

    def close(self):
        self.log_file.close()

    def __del__(self):
        self.close()

    def __end_step(self):
        if self.current_step:
            self.current_step["step"] = self.step
            self.current_step["episode"] = self.episode
            self.buffer.append(json.dumps(self.current_step))
        self.current_step = self.__init_dict()

    @staticmethod
    def __init_dict():
        return defaultdict(lambda: {"times": [], "values": []})

    @staticmethod
    def __init_logging_dir(log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def __reset_episode(self):
        self.__end_step()
        self.episode = 0
        self.step = 0

    def __reset_step(self):
        self.__end_step()
        self.step = 0

    def next_step(self):
        self.__end_step()
        self.step += 1

    def next_episode(self):
        self.__reset_step()
        self.episode += 1

    def write(self):
        self.__buffer_to_file()

    def __buffer_to_file(self):
        self.log_file.write("\n".join(self.buffer))
        self.buffer.clear()

    def is_of_valid_type(self, value: Any) -> bool:
        f"""
        Check if the value of any type in {self.__pretty_valid_types()}
        :param value:
        :return:
        """
        if any(isinstance(value, type) for type in self.valid_types["primitive"]):
            return True

        elif any(isinstance(value, type) for type in self.valid_types["recursive"]):
            value = value.keys() if isinstance(value, dict) else value
            return all(self.is_of_valid_type(sub_value) for sub_value in value)

        else:
            return False

    def log(
        self, key: str, value: Union[Dict, List, Tuple, str, int, float, bool]
    ) -> None:
        f"""
        Write value to list of values for key.
        :param key:
        :param value: the value must of of a type that is
        json serializable. Currently only {self.__pretty_valid_types()} and recursive types of these are
        valid
        :return:
        """
        # TODO add numpy support
        # TODO add instance and benchmark
        if not self.is_of_valid_type(value):
            valid_types = self.__pretty_valid_types()
            raise ValueError(
                f"value {type(value)} is not of valid type or a recursive composition of valid types ({valid_types})"
            )
        self.current_step[key]["times"].append(
            datetime.now().strftime("%d-%m-%y %H:%M:%S")
        )
        self.current_step[key]["values"].append(value)
