"""Container utils."""
from __future__ import annotations

import enum
import json
import socket
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np


class Encoder(json.JSONEncoder):
    """Json Encoder to save tuple and or numpy arrays | numpy floats / integer.

    Adapted from: https://github.com/automl/HPOBench/blob/master/hpobench/util/container_utils.py
    Serializing tuple/numpy array may not work. We need to annotate those types,
    to reconstruct them correctly.
    """

    @staticmethod
    def hint(item):
        """Encode different object types."""
        # Annotate the different item types
        if isinstance(item, tuple):
            return {"__type__": "tuple", "__items__": [Encoder.hint(e) for e in item]}
        if isinstance(item, np.ndarray):
            return {"__type__": "np.ndarray", "__items__": item.tolist()}
        if isinstance(item, np.floating):
            return {"__type__": "np.float", "__items__": float(item)}
        if isinstance(item, np.integer):
            return {"__type__": "np.int32", "__items__": item.tolist()}
        if isinstance(item, enum.Enum):
            return str(item)
        if isinstance(item, gym.Space):
            return Encoder.encode_space(item)
        if isinstance(item, np.dtype):
            return {"__type__": "np.dtype", "__items__": str(item)}

        # If it is a container data structure, go also through the items.
        if isinstance(item, list):
            return [Encoder.hint(e) for e in item]
        if isinstance(item, dict):
            return {key: Encoder.hint(value) for key, value in item.items()}
        return item

    # pylint: disable=arguments-differ
    def encode(self, obj):
        """Generic encode."""
        return super().encode(Encoder.hint(obj))

    @staticmethod
    def encode_space(space_obj: gym.Space):
        """Encode gym space."""
        properties = [
            (
                "__type__",
                ".".join(
                    [space_obj.__class__.__module__, space_obj.__class__.__name__]
                ),
            )
        ]

        if isinstance(
            space_obj,
            gym.spaces.Box
            | gym.spaces.Discrete
            | gym.spaces.MultiDiscrete
            | gym.spaces.MultiBinary,
        ):
            # by default assume all constrcutor arguments are stored under the same name
            # for box we need to drop shape, since either shape or a array for low and
            # height  is required
            __init__ = space_obj.__init__.__func__.__code__
            local_vars = __init__.co_varnames

            # drop self and non-args (self, arg1, arg2, ..., local_var1, local_var2,...)
            arguments = local_vars[1 : __init__.co_argcount]
            attributes_to_serialize = list(
                filter(lambda att: att not in ["shape", "seed"], arguments)
            )

            for attribute in attributes_to_serialize:
                if hasattr(space_obj, attribute):
                    properties.append(
                        (attribute, Encoder.hint(getattr(space_obj, attribute)))
                    )
        elif isinstance(space_obj, gym.spaces.Tuple):
            properties.append(
                ("spaces", [Encoder.encode_space(space) for space in space_obj.spaces])
            )
        elif isinstance(space_obj, gym.spaces.Dict):
            properties.append(
                (
                    "spaces",
                    {
                        name: Encoder.encode_space(space)
                        for name, space in space_obj.spaces.items()
                    },
                )
            )
        else:
            raise NotImplementedError(
                f"Serialisation for type {properties['__type__']} not implemented"
            )

        return dict(properties)


class Decoder(json.JSONDecoder):
    """Adapted from: https://github.com/automl/HPOBench/blob/master/hpobench/util/container_utils.py."""

    def __init__(self, *args, **kwargs):
        """Init decoder."""
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)  # noqa: B026

    def object_hook(self, obj: Any) -> tuple | np.ndarray | float | float | int | Any:
        """Encode different types of objects."""
        if "__type__" in obj:
            __type = obj["__type__"]
            if __type == "tuple":
                return tuple(obj["__items__"])
            if __type == "np.ndarray":
                return np.array(obj["__items__"])
            if __type == "np.float":
                return float(obj["__items__"])
            if __type == "np.int32":
                return np.int32(obj["__items__"])
            if __type == "np.dtype":
                return np.dtype(obj["__items__"])
            if __type.startswith("gymnasium.spaces."):
                return self.decode_space(obj)
        return obj

    def decode_space(self, space_dict: dict) -> gym.Space:
        """Dict to gym space."""
        __type = space_dict["__type__"]
        __class = getattr(gym.spaces, __type.split(".")[-1])

        args = {
            name: value
            for name, value in space_dict.items()
            if name not in ["__type__", "shape"]
        }

        # temporally remove subspace since constructor reseeds them
        if issubclass(__class, gym.spaces.Tuple | gym.spaces.Dict):
            spaces = args["spaces"]
            args["spaces"] = type(args["spaces"])()

        space_object = __class(**args)

        # re-insert afterwards
        if issubclass(__class, gym.spaces.Tuple | gym.spaces.Dict):
            space_object.spaces = spaces

        if isinstance(space_object, gym.spaces.Tuple):
            space_object.spaces = tuple(space_object.spaces)

        print(space_object)
        return space_object


def wait_for_unixsocket(path: str, timeout: float = 10.0) -> None:
    """Wait for a UNIX socket to be created.

    :param path: path to the socket
    :param timeout: timeout in seconds
    :return:

    """
    start = time.time()
    while not Path.exists(path):
        if time.time() - start > timeout:
            raise TimeoutError(
                f"Timeout ({timeout}s) waiting for UNIX socket {path} to be created"
            )
        time.sleep(0.1)


def wait_for_port(port, host="localhost", timeout=5.0):
    """Taken from https://gist.github.com/butla/2d9a4c0f35ea47b7452156c96a4e7b12 -
    Wait until a port starts accepting TCP connections.

    Parameters
    ----------
    port : int
        Port number to check.
    host : str
        Host to check.
    timeout : float
        Timeout in seconds.

    Raises:
    ------
    TimeoutError: The port isn't accepting connection after time specified in `timeout`.

    """
    start_time = time.perf_counter()
    while True:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                break
        except OSError as ex:
            time.sleep(0.01)
            if time.perf_counter() - start_time >= timeout:
                raise TimeoutError(
                    "Waited too long for the port {} on host {} to start accepting "
                    "connections.".format(port, host)
                ) from ex
