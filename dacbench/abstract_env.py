import random

import gym
from gym.utils import seeding
import numpy as np


class AbstractEnv(gym.Env):
    """
    Abstract template for environments
    """

    def __init__(self, config):
        """
        Initialize environment

        Parameters
        -------
        config : dict
            Environment configuration
            If to seed the action space as well
        """
        super(AbstractEnv, self).__init__()
        self.config = config
        if "instance_update_func" in self.config.keys():
            self.instance_updates = self.config["instance_update_func"]
        else:
            self.instance_updates = "round_robin"
        self.instance_set = config["instance_set"]
        self.instance_id_list = sorted(list(self.instance_set.keys()))
        self.instance_index = 0
        self.inst_id = self.instance_id_list[self.instance_index]
        self.instance = self.instance_set[self.inst_id]

        self.test = False
        if "test_set" in self.config.keys():
            self.test_set = config["test_set"]
            self.test_instance_id_list = sorted(list(self.test_set.keys()))
            self.test_instance_index = 0
            self.test_inst_id = self.test_instance_id_list[self.test_instance_index]
            self.test_instance = self.test_set[self.test_inst_id]

            self.training_set = self.instance_set
            self.training_id_list = self.instance_id_list
            self.training_inst_id = self.inst_id
            self.training_instance = self.instance
        else:
            self.test_set = None

        self.benchmark_info = config["benchmark_info"]
        self.initial_seed = None
        self.np_random = None

        self.n_steps = config["cutoff"]
        self.c_step = 0

        self.reward_range = config["reward_range"]

        if "observation_space" in config.keys():
            self.observation_space = config["observation_space"]
        else:
            if not config["observation_space_class"] == "Dict":
                try:
                    self.observation_space = getattr(
                        gym.spaces, config["observation_space_class"]
                    )(
                        *config["observation_space_args"],
                        dtype=config["observation_space_type"],
                    )
                except KeyError:
                    print(
                        "Either submit a predefined gym.space 'observation_space' or an 'observation_space_class' as well as a list of 'observation_space_args' and the 'observation_space_type' in the configuration."
                    )
                    print("Tuple observation_spaces are currently not supported.")
                    raise KeyError

            else:
                try:
                    self.observation_space = getattr(
                        gym.spaces, config["observation_space_class"]
                    )(*config["observation_space_args"])
                except TypeError:
                    print(
                        "To use a Dict observation space, the 'observation_space_args' in the configuration should be a list containing a Dict of gym.Spaces"
                    )
                    raise TypeError

        # TODO: use dicts by default for actions and observations
        # The config could change this for RL purposes
        if "config_space" in config.keys():
            actions = config["config_space"].get_hyperparameters()
            action_types = [type(a).__name__ for a in actions]

            # Uniform action space
            if all(t == action_types[0] for t in action_types):
                if "Float" in action_types[0]:
                    low = np.array([a.lower for a in actions])
                    high = np.array([a.upper for a in actions])
                    self.action_space = gym.spaces.Box(low=low, high=high)
                elif "Integer" in action_types[0] or "Categorical" in action_types[0]:
                    if len(action_types) == 1:
                        try:
                            n = actions[0].upper - actions[0].lower
                        except:
                            n = len(actions[0].choices)
                        self.action_space = gym.spaces.Discrete(n)
                    else:
                        ns = []
                        for a in actions:
                            try:
                                ns.append(a.upper - a.lower)
                            except:
                                ns.append(len(a.choices))
                        self.action_space = gym.spaces.MultiDiscrete(np.array(ns))
                else:
                    raise ValueError(
                        "Only float, integer and categorical hyperparameters are supported as of now"
                    )
            # Mixed action space
            # TODO: implement this
            else:
                raise ValueError("Mixed type config spaces are currently not supported")
        elif "action_space" in config.keys():
            self.action_space = config["action_space"]
        else:
            try:
                self.action_space = getattr(gym.spaces, config["action_space_class"])(
                    *config["action_space_args"]
                )
            except KeyError:
                print(
                    "Either submit a predefined gym.space 'action_space' or an 'action_space_class' as well as a list of 'action_space_args' in the configuration"
                )
                raise KeyError

            except TypeError:
                print("Tuple and Dict action spaces are currently not supported")
                raise TypeError
       
        # seeding the environment after initialising action space
        self.seed(config.get("seed", None), config.get("seed_action_space", False))      

    def step_(self):
        """
        Pre-step function for step count and cutoff

        Returns
        -------
        bool
            End of episode
        """
        done = False
        self.c_step += 1
        if self.c_step >= self.n_steps:
            done = True
        return done

    def reset_(self, instance=None, instance_id=None, scheme=None):
        """
        Pre-reset function for progressing through the instance set
        Will either use round robin, random or no progression scheme
        """
        self.c_step = 0
        if scheme is None:
            scheme = self.instance_updates
        self.use_next_instance(instance, instance_id, scheme=scheme)

    def use_next_instance(self, instance=None, instance_id=None, scheme=None):
        """
        Changes instance according to chosen instance progession

        Parameters
        -------
        instance
            Instance specification for potentional new instances
        instance_id
            ID of the instance to switch to
        scheme
            Update scheme for this progression step (either round robin, random or no progression)
        """
        if instance is not None:
            self.instance = instance
        elif instance_id is not None:
            self.inst_id = instance_id
            self.instance = self.instance_set[self.inst_id]
        elif scheme == "round_robin":
            self.instance_index = (self.instance_index + 1) % len(self.instance_id_list)
            self.inst_id = self.instance_id_list[self.instance_index]
            self.instance = self.instance_set[self.inst_id]
        elif scheme == "random":
            self.inst_id = np.random.choice(self.instance_id_list)
            self.instance = self.instance_set[self.inst_id]

    def step(self, action):
        """
        Execute environment step

        Parameters
        -------
        action
            Action to take

        Returns
        -------
        state
            Environment state
        reward
            Environment reward
        done : bool
            Run finished flag
        info : dict
            Additional metainfo
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset environment

        Returns
        -------
        state
            Environment state
        """
        raise NotImplementedError

    def get_inst_id(self):
        """
        Return instance ID

        Returns
        -------
        int
            ID of current instance
        """
        return self.inst_id

    def get_instance_set(self):
        """
        Return instance set

        Returns
        -------
        list
            List of instances

        """
        return self.instance_set

    def get_instance(self):
        """
        Return current instance

        Returns
        -------
        type flexible
            Currently used instance
        """
        return self.instance

    def set_inst_id(self, inst_id):
        """
        Change current instance ID

        Parameters
        ----------
        inst_id : int
            New instance index
        """
        self.inst_id = inst_id
        self.instance_index = self.instance_id_list.index(self.inst_id)

    def set_instance_set(self, inst_set):
        """
        Change instance set

        Parameters
        ----------
        inst_set: list
            New instance set
        """
        self.instance_set = inst_set
        self.instance_id_list = sorted(list(self.instance_set.keys()))

    def set_instance(self, instance):
        """
        Change currently used instance

        Parameters
        ----------
        instance:
            New instance
        """
        self.instance = instance

    def seed_action_space(self, seed=None):
        """
        Seeds the action space.
        Parameters
        ----------
        seed : int, default None
            if None self.initial_seed is be used

        Returns
        -------

        """
        if seed is None:
            seed = self.initial_seed

        self.action_space.seed(seed)

    def seed(self, seed=None, seed_action_space=False):
        """
        Set rng seed

        Parameters
        ----------
        seed:
            seed for rng
        seed_action_space: bool, default False
            if to seed the action space as well
        """

        self.initial_seed = seed
        # maybe one should use the seed generated by seeding.np_random(seed) but it can be to large see issue https://github.com/openai/gym/issues/2210
        random.seed(seed)
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        # uses the uncorrelated seed from seeding but makes sure that no randomness is introduces.

        if seed_action_space:
            self.seed_action_space()

        return [seed]

    def use_test_set(self):
        """
        Change to test instance set
        """
        if self.test_set is None:
            raise ValueError(
                "No test set was provided, please check your benchmark config."
            )

        self.test = True
        self.training_set = self.instance_set
        self.training_id_list = self.instance_id_list
        self.training_inst_id = self.inst_id
        self.training_instance = self.instance

        self.instance_set = self.test_set
        self.instance_id_list = self.test_instance_id_list
        self.inst_id = self.test_inst_id
        self.instance = self.test_instance

    def use_training_set(self):
        """
        Change to training instance set
        """
        self.test = False
        self.test_set = self.instance_set
        self.test_instance_id_list = self.instance_id_list
        self.test_inst_id = self.inst_id
        self.test_instance = self.instance

        self.instance_set = self.training_set
        self.instance_id_list = self.training_id_list
        self.inst_id = self.training_inst_id
        self.instance = self.training_instance
