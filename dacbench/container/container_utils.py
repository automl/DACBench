import json
import numpy as np
import enum

from typing import Any, Union, Tuple, List

from hpobench.util.rng_helper import serialize_random_state, deserialize_random_state


class Encoder(json.JSONEncoder):
    """ Json Encoder to save tuple and or numpy arrays | numpy floats / integer.
    Adapted from: https://github.com/automl/HPOBench/blob/master/hpobench/util/container_utils.py
    Serializing tuple/numpy array may not work. We need to annotate those types, to reconstruct them correctly.
    """
    # pylint: disable=arguments-differ
    def encode(self, obj):
        def hint(item):
            # Annotate the different item types
            if isinstance(item, tuple):
                return {'__type__': 'tuple', '__items__': [hint(e) for e in item]}
            if isinstance(item, np.ndarray):
                return {'__type__': 'np.ndarray', '__items__': item.tolist()}
            if isinstance(item, np.floating):
                return {'__type__': 'np.float', '__items__': float(item)}
            if isinstance(item, np.integer):
                return {'__type__': 'np.int', '__items__': item.tolist()}
            if isinstance(item, enum.Enum):
                return str(item)
            if isinstance(item, np.random.RandomState):
                rs = serialize_random_state(item)
                return {'__type__': 'random_state', '__items__': rs}

            # If it is a container data structure, go also through the items.
            if isinstance(item, list):
                return [hint(e) for e in item]
            if isinstance(item, dict):
                return {key: hint(value) for key, value in item.items()}
            return item

        return super(Encoder, self).encode(hint(obj))


class Decoder(json.JSONDecoder):
    """
    Adapted from: https://github.com/automl/HPOBench/blob/master/hpobench/util/container_utils.py

    """
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj: Any) -> Union[Union[tuple, np.ndarray, float, float, int], Any]:
        if '__type__' in obj:
            __type = obj['__type__']

            if __type == 'tuple':
                return tuple(obj['__items__'])
            if __type == 'np.ndarray':
                return np.array(obj['__items__'])
            if __type == 'np.float':
                return np.float(obj['__items__'])
            if __type == 'np.int':
                return np.int(obj['__items__'])
            if __type == 'random_state':
                return deserialize_random_state(obj['__items__'])
        return obj


def deserialize_random_state(random_state: Tuple[int, List, int, int, int]) -> np.random.RandomState:
    (rnd0, rnd1, rnd2, rnd3, rnd4) = random_state
    rnd1 = [np.uint32(number) for number in rnd1]
    random_state = np.random.RandomState()
    random_state.set_state((rnd0, rnd1, rnd2, rnd3, rnd4))
    return random_state



