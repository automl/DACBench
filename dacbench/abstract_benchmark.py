import json
from types import FunctionType
from typing import Callable

import numpy as np
from gym import spaces
from functools import partial
from dacbench import wrappers

# from dacbench import ModuleLogger



class AbstractBenchmark:
    """
    Abstract template for benchmark classes
    """

    def __init__(self, config_path=None, config:  'objdict'=None):
        """
        Initialize benchmark class

        Parameters
        -------
        config_path : str
            Path to load configuration from (if read from file)
        config : objdict
            Object dict containing the config
        """
        if config is not None and config_path is not None:
            raise ValueError("Both path to config and config where provided")
        self.wrap_funcs = []
        if config_path:
            self.config_path = config_path
            self.read_config_file(self.config_path)
        elif config:
            self.load_config(config)
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

    def serialize_config(self):
        """
        Save configuration to json

        Parameters
        ----------
        path : str
            File to save config to
        """
        conf = self.config.copy()
        if "observation_space_type" in self.config:
            conf["observation_space_type"] = f"{self.config['observation_space_type']}"
            if isinstance(conf["observation_space_args"][0], dict):
                conf["observation_space_args"] = self.jsonify_dict_space(
                    conf["observation_space_args"][0]
                )
        elif "observation_space" in self.config:
            conf["observation_space"] = self.space_to_list(conf["observation_space"])

        if "action_space" in self.config:
            conf["action_space"] = self.space_to_list(conf["action_space"])

        conf = AbstractBenchmark.__stringify_functions(conf)

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
                elif isinstance(conf[k], np.ndarray):
                    conf[k] = conf[k].tolist()

        conf["wrappers"] = self.jsonify_wrappers()

        # can be recovered from instance_set_path, and could contain function that are not serializable
        if "instance_set" in conf:
            del conf["instance_set"]

        return conf

    @staticmethod
    def from_json(json_config):
        config = objdict(json.loads(json_config))
        return AbstractBenchmark(config=config)
    
    def to_json(self):
        conf = self.serialize_config()
        return json.dumps(conf)

    def save_config(self, path):
        conf = self.serialize_config()
        with open(path, "w") as fp:
            json.dump(conf, fp)

    def jsonify_wrappers(self):
        wrappers = []
        for func in self.wrap_funcs:
            args = func.args
            arg_descriptions = []
            contains_func = False
            func_dict = {}
            for i in range(len(args)):
                if callable(args[i]):
                    contains_func = True
                    func_dict[f"{args[i]}"] = [args[i].__module__, args[i].__name__]
                    arg_descriptions.append(["function", f"{args[i]}"])
                # elif isinstance(args[i], ModuleLogger):
                #    pass
                else:
                    arg_descriptions.append({args[i]})
            function = func.func.__name__
            if contains_func:
                wrappers.append([function, arg_descriptions, func_dict])
            else:
                wrappers.append([function, arg_descriptions])
        return wrappers

    def dejson_wrappers(self, wrapper_list):
        for i in range(len(wrapper_list)):
            import importlib

            func = getattr(wrappers, wrapper_list[i][0])
            arg_descriptions = wrapper_list[i][1]
            args = []
            for a in arg_descriptions:
                if a[0] == "function":
                    module = importlib.import_module(wrapper_list[i][2][a[1]][0])
                    name = wrapper_list[i][2][a[1]][0]
                    func = getattr(module, name)
                    args.append(func)
                # elif a[0] == "logger":
                #    pass
                else:
                    args.append(a)

            self.wrap_funcs.append(partial(func, *args))

    @staticmethod
    def __import_from(module : str, name : str):
        """
        Imports the class / function / ... with name from module
        Parameters
        ----------
        module
        name

        Returns
        -------
        the imported object
        """
        module = __import__(module, fromlist=[name])
        return getattr(module, name)

    @staticmethod
    def __decorate_config_with_functions(conf : dict):
        """
        Replaced the stringified functions with the callable objects
        Parameters
        ----------
        config

        Returns
        -------

        """
        for key, value in {k: v for k, v in conf.items() if isinstance(v, list) and len(v) == 3 and v[0] == 'function'}.items():
            _, module_name, function_name = value
            conf[key] = AbstractBenchmark.__import_from(module_name, function_name)
        return conf

    @staticmethod
    def __stringify_functions(conf : dict) -> dict:
        """
        Replaced all callables in the config with a
        triple ('function', module_name, function_name)

        Parameters
        ----------
        config

        Returns
        -------
        modified dict
        """
        for key, value in {k: v for k, v in conf.items() if isinstance(v, FunctionType)}.items():
            conf[key] = [
                'function',
                conf[key].__module__,
                conf[key].__name__
            ]
        return conf

    def space_to_list(self, space):
        res = []
        if isinstance(space, spaces.Box):
            res.append("Box")
            res.append([space.low.tolist(), space.high.tolist()])
            res.append("numpy.float32")
        elif isinstance(space, spaces.Discrete):
            res.append("Discrete")
            res.append([space.n])
        elif isinstance(space, spaces.Dict):
            res.append("Dict")
            res.append(self.jsonify_dict_space(space.spaces))
        elif isinstance(space, spaces.MultiDiscrete):
            res.append("MultiDiscrete")
            res.append([space.nvec])
        elif isinstance(space, spaces.MultiBinary):
            res.append("MultiBinary")
            res.append([space.n])
        return res

    def list_to_space(self, space_list):
        if space_list[0] == "Dict":
            args = self.dictify_json(space_list[1])
            space = getattr(spaces, space_list[0])(args)
        elif len(space_list) == 2:
            space = getattr(spaces, space_list[0])(*space_list[1])
        else:
            typestring = space_list[2].split(".")[1]
            dt = getattr(np, typestring)
            args = [np.array(arg) for arg in space_list[1]]
            space = getattr(spaces, space_list[0])(*args, dtype=dt)
        return space

    def jsonify_dict_space(self, dict_space):
        keys = []
        arguments = []
        for k in dict_space.keys():
            keys.append(k)
            value = dict_space[k]
            if not isinstance(value, (spaces.Box, spaces.Discrete)):
                raise ValueError(f"Only Dict spaces made up of Box spaces or discrete spaces are supported but got {type(value)}")

            if isinstance(value, spaces.Box):
                low = value.low.tolist()
                high = value.high.tolist()
                arguments.append([low, high])

            if isinstance(value, spaces.Discrete):
                n = value.n
                arguments.append([n])
        return [keys, arguments]

    def dictify_json(self, dict_list):
        dict_space = {}
        types, args = dict_list
        for type, args_ in zip(types, args):
            prepared_args = map(np.array, args_)
            if type == "box":
                dict_space[type] = spaces.Box(*prepared_args, dtype=np.float32)
            elif type == "discrete":
                dict_space[type] = spaces.Discrete(*prepared_args)
            else:
                raise TypeError(f"Currently only Discrete and Box spaces are allowed in Dict spaces got {type}")

        return dict_space

    def load_config(self, config :  'objdict'):
        self.config = config
        if "observation_space_type" in self.config:
            # Types have to be numpy dtype (for gym spaces)s
            if type(self.config["observation_space_type"]) == str:
                if self.config["observation_space_type"] == "None":
                    self.config["observation_space_type"] = None
                else:
                    typestring = self.config["observation_space_type"].split(" ")[1][
                        :-2
                    ]
                    typestring = typestring.split(".")[1]
                    self.config["observation_space_type"] = getattr(np, typestring)
        if "observation_space" in self.config:
            self.config["observation_space"] = self.list_to_space(
                self.config["observation_space"]
            )
        elif "observation_space_class" == "Dict":
            self.config["observation_space_args"] = self.dictify_json(
                self.config["observation_space_args"]
            )

        if "action_space" in self.config:
            self.config["action_space"] = self.list_to_space(
                self.config["action_space"]
            )

        if "wrappers" in self.config:
            self.dejson_wrappers(self.config["wrappers"])
            del self.config["wrappers"]

        self.config = AbstractBenchmark.__decorate_config_with_functions(self.config)

        for k in self.config.keys():
            if type(self.config[k]) == list:
                if type(self.config[k][0]) == list:
                    map(np.array, self.config[k])
                self.config[k] = np.array(self.config[k])

    def read_config_file(self, path):
        """
        Read configuration from file

        Parameters
        ----------
        path : str
            Path to config file
        """
        with open(path, "r") as fp:
            config = objdict(json.load(fp))
        
        self.load_config(config)
    

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
        kind : str
            Name of observation space class
        args : list
            List of arguments to pass to observation space class
        data_type : type
            Data type of observation space
        """
        self.config["observation_space"] = kind
        self.config["observation_space_args"] = args
        self.config["observation_space_type"] = data_type

    def register_wrapper(self, wrap_func):
        if isinstance(wrap_func, list):
            self.wrap_funcs.append(*wrap_func)
        else:
            self.wrap_funcs.append(wrap_func)


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

    def copy(self):
        return objdict(**super().copy())
