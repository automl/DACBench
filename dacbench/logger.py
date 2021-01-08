import json
from abc import ABCMeta, abstractmethod
from collections import defaultdict, ChainMap
from datetime import datetime
from functools import reduce
from itertools import chain
from numbers import Number
from pathlib import Path
from typing import Union, Dict, Any, Tuple, List

import numpy as np
import pandas as pd

from dacbench import AbstractEnv, AbstractBenchmark
from dacbench.abstract_agent import AbstractDACBenchAgent


def load_logs(log_file: Path):
    with open(log_file, "r") as log_file:
        logs = list(map(json.loads, log_file))

    return logs


def split(predicate, iterable) -> Tuple[List, List]:
    """
    Splits the iterable into two list depending on the result of predicate.
    :param predicate:
    :param iterable:
    :return: positives, negatives
    """
    positives, negatives = [], []

    for item in iterable:
        (positives if predicate(item) else negatives).append(item)

    return positives, negatives


def flatten_log_entry(log_entry: Dict) -> List[Dict]:
    """
    Transforms a log entry of format like
    {
        'step': 0,
        'episode': 2,
        'some_value': {
            'values' : [34, 45],
            'times':['28-12-20 16:20:53', '28-12-20 16:21:30'],
        }
    }
    into
    [
        { 'step': 0,'episode': 2, 'value': 34, 'time': '28-12-20 16:20:53'},
        { 'step': 0,'episode': 2, 'value': 45, 'time': '28-12-20 16:21:30'}
    ]

    :param log_entry:
    :return:
    """
    dict_entries, top_level_entries = split(
        lambda item: isinstance(item[1], dict), log_entry.items()
    )
    rows = []
    for value_name, value_dict in dict_entries:
        current_rows = (
            dict(
                top_level_entries
                + [("value", value), ("time", time), ("name", value_name)]
            )
            for value, time in zip(value_dict["values"], value_dict["times"])
        )

        rows.extend(map(dict, current_rows))

    return rows


def log2dataframe(logs: List[dict], wide=False, include_time=False) -> pd.DataFrame:
    """
    Converts a list of log entries to a pandas dataframe.
    :param logs:
    :param wide: determines the format of the dataframe.
    wide=False (default) produces a dataframe with columns (episode, step, time, name, value)
    wide=True returns a dataframe (episode, step, time, name_1, name_2, ...) if the variable name_n has not been logged
    at (episode, step, time) name_n is NaN.
    :param include_time: drops the time columns mostly used in combination with wide=True to reduce NaN values
    :return:
    """
    flat_logs = map(flatten_log_entry, logs)
    rows = reduce(lambda l1, l2: l1 + l2, flat_logs)

    dataframe = pd.DataFrame(rows)
    dataframe.time = pd.to_datetime(dataframe.time)

    if not include_time:
        dataframe = dataframe.drop(columns=["time"])

    if wide:
        primary_index_columns = ["episode", "step"]
        field_id_column = "name"
        additional_columns = list(
            set(dataframe.columns)
            - set(primary_index_columns + ["time", "value", field_id_column])
        )
        index_columns = primary_index_columns + additional_columns + [field_id_column]
        dataframe = dataframe.set_index(index_columns)
        dataframe = dataframe.unstack()
        dataframe.reset_index(inplace=True)
        dataframe.columns = [a if b == "" else b for a, b in dataframe.columns]

    return dataframe


class AbstractLogger(metaclass=ABCMeta):
    """
    Logger interface.

    The logger classes provide a way of writing structured logs as jsonl files and also help to track information like
    current episode, step, time ...

    In the jsonl log file each row corresponds to a step.
    """

    valid_types = {
        "recursive": [dict, list, tuple, np.ndarray],
        "primitive": [str, int, float, bool, np.number],
    }

    def __init__(
        self,
        experiment_name: str,
        output_path: Path,
        step_write_frequency: int = None,
        episode_write_frequency: int = 1,
    ):
        """
        :param experiment_name:
        :param output_path:
        :param step_write_frequency: number of steps after which the loggers writes to file.
        If None only the data is only written to file if  write is called, if triggered by episode_write_frequency
        or on close
        :param episode_write_frequency: see step_write_frequency
        """
        self.experiment_name = experiment_name
        self.output_path = output_path
        self.log_dir = self._init_logging_dir(self.output_path / self.experiment_name)
        self.step_write_frequency = step_write_frequency
        self.episode_write_frequency = episode_write_frequency
        self.additional_info = {"instance": None}

    def _pretty_valid_types(self) -> str:
        """
        Returns a string pretty string representation of the types that can be logged as values
        :return:
        """
        valid_types = chain(
            self.valid_types["recursive"], self.valid_types["primitive"]
        )
        return ", ".join(map(lambda type_: type_.__name__, valid_types))

    @staticmethod
    def _init_logging_dir(log_dir: Path):
        """
        Prepares the logging directory
        :param log_dir:
        :return:
        """
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def is_of_valid_type(self, value: Any) -> bool:
        f"""
        Check if the value of any type in {self._pretty_valid_types()}
        :param value:
        :return:
        """
        if any(isinstance(value, type) for type in self.valid_types["primitive"]):
            return True

        elif any(isinstance(value, type) for type in self.valid_types["recursive"]):
            value = value.vlaues() if isinstance(value, dict) else value
            return all(self.is_of_valid_type(sub_value) for sub_value in value)

        else:
            return False

    @abstractmethod
    def close(self):
        """
        Makes sure, that all remaining entries in the are written to file and the file is closed
        :return:
        """
        pass

    @abstractmethod
    def next_step(self):
        """
        Call at the end of the step.
        Updates the internal state and dumps the information of the last step into a json
        :return:
        """
        pass

    @abstractmethod
    def next_episode(self):
        """
        Call at the end of episode.

        See next_step
        :return:
        """
        pass

    @abstractmethod
    def write(self):
        """
        Writes buffered logs to file
        :return:
        """
        pass

    @abstractmethod
    def log(self, key, value, **kwargs):
        f"""
        Writes value to list of values for key.
        :param key:
        :param value: the value must of of a type that is
        json serializable. Currently only {self._pretty_valid_types()} and recursive types of these are
        valid
        :return:
        """
        pass

    @abstractmethod
    def log_dict(self, data):
        """
        Alternative to log if more the one value should be logged
        :param data: a dict with key-value so that each value is a valid value for log
        :return:
        """
        pass

    @abstractmethod
    def log_space(self, key, value, space_info=None):
        """
        Special for logging gym.spaces.
        Currently three types are supported:
        * Numbers: e.g. samples from Discrete
        * Fixed length arrays like MultiDiscrete or Box
        * Dict: assuming each key has fixed length array
        :param space_info: a list of column names. The length of this list must equal the resulting number of columns.
        :param key:
        :param value:
        :return:
        """
        pass


class ModuleLogger(AbstractLogger):
    """
    A logger for handling logging of one module. e.g. a wrapper or toplevel general logging.

    Don't create manually use Logger to manage ModuleLoggers
    """

    def __init__(
        self,
        output_path: Path,
        experiment_name: str,
        module: str,
        step_write_frequency: int = None,
        episode_write_frequency: int = 1,
    ) -> None:
        """
        All results are placed under 'output_path / experiment_name'
        :param output_path: the path where logged information should be stored
        :param experiment_name: name of the experiment.
        :param the module (mostly name of the wrapper), each wrapper get's its own file
        :param step_write_frequency: number of steps after which the loggers writes to file.
         If None only the data is only written to file if  write is called, if triggered by episode_write_frequency
        or on close
        :param episode_write_frequency: see step_write_frequency
        """

        super(ModuleLogger, self).__init__(
            experiment_name, output_path, step_write_frequency, episode_write_frequency
        )

        self.log_file = open(self.log_dir / f"{module}.jsonl", "w")

        self.step = 0
        self.episode = 0
        self.buffer = []
        self.current_step = self.__init_dict()

    def get_logfile(self) -> Path:
        return Path(self.log_file.name)

    def close(self):
        if not self.log_file.closed:
            self.write()
            self.log_file.close()

    def __del__(self):
        if not self.log_file.closed:
            self.close()

    @staticmethod
    def __json_default(object):
        if isinstance(object, np.ndarray):
            return object.tolist()
        elif isinstance(object, np.number):
            return object.item()
        else:
            raise ValueError(f"Type {type(object)} not supported")

    def __end_step(self):
        if self.current_step:
            self.current_step["step"] = self.step
            self.current_step["episode"] = self.episode
            self.current_step.update(self.additional_info)
            self.buffer.append(
                json.dumps(self.current_step, default=self.__json_default)
            )
        self.current_step = self.__init_dict()

    @staticmethod
    def __init_dict():
        return defaultdict(lambda: {"times": [], "values": []})

    def reset_episode(self):
        self.__end_step()
        self.episode = 0
        self.step = 0

    def __reset_step(self):
        self.__end_step()
        self.step = 0

    def next_step(self):
        self.__end_step()
        if (
            self.step_write_frequency is not None
            and self.step % self.step_write_frequency == 0
        ):
            self.write()
        self.step += 1

    def next_episode(self):
        self.__reset_step()
        if (
            self.episode_write_frequency is not None
            and self.episode % self.episode_write_frequency == 0
        ):
            self.write()
        self.episode += 1

    def write(self):
        self.__end_step()
        self.__buffer_to_file()

    def __buffer_to_file(self):
        if len(self.buffer) > 0:
            self.log_file.write("\n".join(self.buffer))
            self.log_file.write("\n")
            self.buffer.clear()
            self.log_file.flush()

    def set_additional_info(self, **kwargs):
        """
        Can be used to log additional infomation for each step e.g. for seed, and instance id.
        :param kwargs:
        :return:
        """
        self.additional_info.update(kwargs)

    def log(
        self, key: str, value: Union[Dict, List, Tuple, str, int, float, bool], **kwargs
    ) -> None:
        f"""
        Write value to list of values for key.
        :param **kwargs:
        :param key:
        :param value: the value must of of a type that is
        json serializable. Currently only {self._pretty_valid_types()} and recursive types of these are
        valid
        :return:
        """
        self.__log(key, value, datetime.now().strftime("%d-%m-%y %H:%M:%S.%f"))

    def __log(self, key, value, time):
        if not self.is_of_valid_type(value):
            valid_types = self._pretty_valid_types()
            raise ValueError(
                f"value {type(value)} is not of valid type or a recursive composition of valid types ({valid_types})"
            )
        self.current_step[key]["times"].append(time)
        self.current_step[key]["values"].append(value)

    def log_dict(self, data):
        time = datetime.now().strftime("%d-%m-%y %H:%M:%S.%f")
        for key, value in data.items():
            self.__log(key, value, time)

    @staticmethod
    def __space_dict(key, value, space_info):
        if isinstance(value, np.ndarray) and len(value.shape) == 0:
            value = value.item()

        if isinstance(value, Number):
            if space_info is None:
                data = {key: value}
            else:
                if len(space_info) != 1:
                    raise ValueError(
                        f"Space info must match length (expect 1 != got{len(space_info)}"
                    )

                data = {f"{key}_{space_info[0]}": value}

        elif isinstance(value, np.ndarray):
            if space_info is not None and len(space_info) != len(value):
                raise ValueError(
                    f"Space info must match length (expect {len(value)} != got{len(space_info)}"
                )
            key_suffix = (
                enumerate(value) if space_info is None else zip(space_info, value)
            )
            data = {f"{key}_{suffix}": x for suffix, x in key_suffix}

        elif isinstance(value, dict):
            key_suffix = (
                value.items() if space_info is None else zip(space_info, value.values())
            )
            dicts = (
                ModuleLogger.__space_dict(f"{key}_{sub_key}", sub_value, None)
                for sub_key, sub_value in key_suffix
            )
            data = dict(ChainMap(*dicts))
        else:
            raise ValueError("Space does not seem be supported")

        return data

    def log_space(self, key, value, space_info=None):
        data = self.__space_dict(key, value, space_info)
        self.log_dict(data)


class Logger(AbstractLogger):
    """
    A logger that manages the creation of the module loggers.

    To get a ModuleLogger for you module (e.g. wrapper) call module_logger = Logger(...).add_module("my_wrapper").
    From now on  module_logger.log(...) or logger.log(..., module="my_wrapper") can be used to log.

    The logger module takes care of updating information like episode and step in the subloggers. To indicate to the loggers
    the end of the episode or the next_step simple call logger.next_episode() or logger.next_step().
    """

    def __init__(
        self,
        experiment_name: str,
        output_path: Path,
        step_write_frequency: int = None,
        episode_write_frequency: int = 1,
    ) -> None:
        """
        :param experiment_name:
        :param output_path:
        :param step_write_frequency: number of steps after which the loggers writes to file.
        If None only the data is only written to file if  write is called, if triggered by episode_write_frequency
        or on close
        :param episode_write_frequency: see step_write_frequency
        """
        super(Logger, self).__init__(
            experiment_name, output_path, step_write_frequency, episode_write_frequency
        )
        self.env: AbstractEnv = None
        self.module_logger: Dict[str, ModuleLogger] = dict()

    def close(self):
        for _, module_logger in self.module_logger.items():
            module_logger.close()

    def __del__(self):
        self.close()

    def next_step(self):
        for _, module_logger in self.module_logger.items():
            module_logger.next_step()

    def next_episode(self):
        for _, module_logger in self.module_logger.items():
            module_logger.next_episode()

        self.__update_auto_additional_info()

    def __update_auto_additional_info(self):
        # TODO add seed too if av?
        self.set_additional_info(instance=self.env.get_inst_id())

    def reset_episode(self):
        for _, module_logger in self.module_logger.items():
            module_logger.reset_episode()

    def write(self):
        for _, module_logger in self.module_logger.items():
            module_logger.write()

    def add_module(self, module: Union[str, type]) -> ModuleLogger:
        """
        Creates a sub-logger. For more details see class level documentation
        :param module:
        :return: sub-logger for the given module
        """
        if isinstance(module, str):
            pass
        elif isinstance(module, type):
            module = module.__name__
        else:
            module = module.__class__

        if module in self.module_logger:
            raise ValueError(f"Module {module} already registered")
        else:
            self.module_logger[module] = ModuleLogger(
                self.output_path,
                self.experiment_name,
                module,
                self.step_write_frequency,
                self.episode_write_frequency,
            )
            if self.env is not None:
                self.module_logger[module].set_additional_info(
                    instance=self.env.get_inst_id()
                )

        return self.module_logger[module]

    def add_agent(self, agent: AbstractDACBenchAgent):
        """
        Writes information about the agent
        :param agent:
        :return:
        """
        agent_config = {"type": agent.__class__}
        with open(self.output_path / "agent.json") as f:
            json.dump(agent_config, f)

    def set_env(self, env: AbstractEnv):
        """
        Writes information about the env
        :param env:
        :return:
        """
        self.env = env
        self.__update_auto_additional_info()

    def add_benchmark(self, benchmark: AbstractBenchmark) -> None:
        """
        Writes the config to the experiment path
        :param benchmark:
        :return:
        """
        benchmark.save_config(self.output_path)

    def set_additional_info(self, **kwargs):
        for _, module_logger in self.module_logger.items():
            module_logger.set_additional_info(**kwargs)

    def log(self, key, value, module):
        if module not in self.module_logger:
            raise ValueError(f"Module {module} not registered yet")
        self.module_logger.log(key, value)

    def log_space(self, key, value, module, space_info=None):
        if module not in self.module_logger:
            raise ValueError(f"Module {module} not registered yet")
        self.module_logger.log_space(key, value, space_info)

    def log_dict(self, data, module):
        if module not in self.module_logger:
            raise ValueError(f"Module {module} not registered yet")
        self.module_logger.log_space(data)
