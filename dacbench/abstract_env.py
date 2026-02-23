"""Abstract Environment."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding


class AbstractEnv(ABC, gym.Env):
    """Abstract template for environments."""

    def __init__(self, config):
        """Initialize environment.

        Parameters
        ----------
        config : dict
            Environment configuration
            If to seed the action space as well

        """
        super().__init__()
        self.config = config
        self.instance_updates = self.config.get("instance_update_func", "round_robin")
        self.instance_set = config["instance_set"]
        self.instance_id_list = sorted(self.instance_set.keys())
        self.instance_index = 0
        self.inst_id = self.instance_id_list[self.instance_index]
        self.instance = self.instance_set[self.inst_id]

        self.test = False
        if "test_set" in self.config:
            self.test_set = config["test_set"]
            self.test_instance_id_list = sorted(self.test_set.keys())
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

        if "observation_space" in config:
            self.observation_space = config["observation_space"]
        elif config["observation_space_class"] != "Dict":
            try:
                self.observation_space = getattr(
                    gym.spaces, config["observation_space_class"]
                )(
                    *config["observation_space_args"],
                    dtype=config["observation_space_type"],
                )
            except KeyError as err:
                print(
                    "Either submit a predefined gym.space 'observation_space' or an "
                    "'observation_space_class' as well as a list of "
                    "'observation_space_args' and the 'observation_space_type' "
                    "in the configuration."
                )
                print("Tuple observation_spaces are currently not supported.")
                raise KeyError from err

        else:
            try:
                self.observation_space = getattr(
                    gym.spaces, config["observation_space_class"]
                )(*config["observation_space_args"])
            except AssertionError as err:
                print(
                    "To use a Dict observation space, the 'observation_space_args' in "
                    "the configuration should be a list containing a Dict of gym.Spaces"
                )
                raise TypeError from err

        # TODO: use dicts by default for actions and observations
        # The config could change this for RL purposes
        if "config_space" in config:
            actions = list(config["config_space"].values())
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
                            n = actions[0].upper - actions[0].lower + 1
                        except:  # noqa: E722
                            n = len(actions[0].choices)
                        self.action_space = gym.spaces.Discrete(n)
                    else:
                        ns = []
                        for a in actions:
                            try:
                                ns.append(a.upper - a.lower + 1)
                            except:  # noqa: E722
                                ns.append(len(a.choices))
                        self.action_space = gym.spaces.MultiDiscrete(np.array(ns))
                else:
                    raise ValueError(
                        "Only float, integer and categorical hyperparameters "
                        "are supported as of now"
                    )
            # Mixed action space
            else:
                subspaces = {}
                for t, a in zip(action_types, actions, strict=False):
                    if "Float" in t:
                        subspaces[a.name] = gym.spaces.Box(
                            low=a.lower, high=a.upper, dtype=np.float32
                        )
                    elif "Integer" in t or "Categorical" in t:
                        try:
                            n = a.upper - a.lower
                        except:  # noqa: E722
                            n = len(a.choices)
                        subspaces[a.name] = gym.spaces.Discrete(n)
                    else:
                        raise ValueError(
                            "Only float, integer and categorical hyperparameters "
                            "are supported as of now"
                        )
                self.action_space = gym.spaces.Dict(subspaces)
        elif "action_space" in config:
            self.action_space = config["action_space"]
        else:
            try:
                self.action_space = getattr(gym.spaces, config["action_space_class"])(
                    *config["action_space_args"]
                )
            except KeyError as err:
                print(
                    "Either submit a predefined gym.space 'action_space' or an "
                    "'action_space_class' as well as a list of 'action_space_args'"
                    " in the configuration"
                )
                raise KeyError from err

            except TypeError as err:
                print("Tuple and Dict action spaces are currently not supported")
                raise TypeError from err

        # seeding the environment after initialising action space
        self.seed(config.get("seed", None), config.get("seed_action_space", False))

    def step_(self):
        """Pre-step function for step count and cutoff.

        Returns:
        --------
        bool: End of episode

        """
        truncated = False
        self.c_step += 1
        if self.c_step >= self.n_steps:
            truncated = True
        return truncated

    def reset_(
        self, seed=0, options=None, instance=None, instance_id=None, scheme=None
    ):
        """Pre-reset function for progressing through the instance set.
        Will either use round robin, random or no progression scheme.
        """
        if options is None:
            options = {}
        if seed is not None:
            self.seed(seed, self.config.get("seed_action_space", False))
        self.c_step = 0
        if scheme is None:
            scheme = options.get("scheme", self.instance_updates)
        if instance is None:
            instance = options.get("instance")
        if instance_id is None:
            instance_id = options.get("instance_id")
        self.use_next_instance(instance, instance_id, scheme=scheme)

    def use_next_instance(self, instance=None, instance_id=None, scheme=None):
        """Changes instance according to chosen instance progession.

        Parameters
        ----------
        instance
            Instance specification for potentional new instances
        instance_id
            ID of the instance to switch to
        scheme
            Update scheme for this progression step
            (either round robin, random or no progression)

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
            rng = np.random.default_rng()
            self.inst_id = rng.choice(self.instance_id_list)
            self.instance = self.instance_set[self.inst_id]

    @abstractmethod
    def step(self, action):
        """Execute environment step.

        Parameters
        ----------
        action
            Action to take

        Returns:
        --------
        state
            Environment state
        reward
            Environment reward
        terminated: bool
            Run finished flag
        truncated: bool
            Run timed out flag
        info : dict
            Additional metainfo

        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, seed: int | None = None):
        """Reset environment.

        Parameters
        ----------
        seed
            Seed for the environment

        Returns:
        --------
        state
            Environment state
        info: dict
            Additional metainfo

        """
        raise NotImplementedError

    def get_inst_id(self):
        """Return instance ID.

        Returns:
        --------
        int: ID of current instance
        """
        return self.inst_id

    def get_instance_set(self):
        """Return instance set.

        Returns:
        --------
        list: List of instances
        """
        return self.instance_set

    def get_instance(self):
        """Return current instance.

        Returns:
        --------
        type flexible: Currently used instance
        """
        return self.instance

    def set_inst_id(self, inst_id):
        """Change current instance ID.

        Parameters
        ----------
        inst_id : int
            New instance index

        """
        self.inst_id = inst_id
        self.instance_index = self.instance_id_list.index(self.inst_id)

    def set_instance_set(self, inst_set):
        """Change instance set.

        Parameters
        ----------
        inst_set: list
            New instance set

        """
        self.instance_set = inst_set
        self.instance_id_list = sorted(self.instance_set.keys())

    def set_instance(self, instance):
        """Change currently used instance.

        Parameters
        ----------
        instance:
            New instance

        """
        self.instance = instance

    def seed(self, seed=None, seed_action_space=False):
        """Set rng seed.

        Parameters
        ----------
        seed:
            seed for rng
        seed_action_space: bool, default False
            if to seed the action space as well

        """
        if seed is None:
            rng = np.random.default_rng()
            seed = rng.integers(0, 2**32 - 1)
        self.initial_seed = seed
        # maybe one should use the seed generated by seeding.np_random(seed)
        # but it can be to large see issue https://github.com/openai/gym/issues/2210
        rng = np.random.default_rng(seed)
        random.seed(int(seed))
        self.np_random, seed = seeding.np_random(int(seed))
        # uses the uncorrelated seed from seeding but makes sure that no
        # randomness is introduced.

        if seed_action_space:
            self.action_space.seed(seed)

        return [seed]

    def use_test_set(self):
        """Change to test instance set."""
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
        """Change to training instance set."""
        self.test = False
        self.test_set = self.instance_set
        self.test_instance_id_list = self.instance_id_list
        self.test_inst_id = self.inst_id
        self.test_instance = self.instance

        self.instance_set = self.training_set
        self.instance_id_list = self.training_id_list
        self.inst_id = self.training_inst_id
        self.instance = self.training_instance


class AbstractMADACEnv(AbstractEnv):
    """Multi-Agent version of DAC environment."""

    def __init__(self, config):
        """Initialize environment.

        Parameters
        ----------
        config : dict
            Environment configuration
            If to seed the action space as well

        """
        super().__init__(config)
        self.multi_agent = False
        if "multi_agent" in config:
            self.multi_agent = config.multi_agent

        if self.multi_agent:
            space_class = type(self.action_space)
            if space_class == gym.spaces.Box:
                num_hps = len(self.action_space.low)
            elif space_class == gym.spaces.MultiDiscrete:
                num_hps = len(self.action_space.nvec)
            elif space_class == gym.spaces.Dict:
                num_hps = len(self.action_space.spaces)
            else:
                print(
                    "The MultiAgent environment currently only supports "
                    "action spaces of types Box, Dict and MultiDiscrete"
                )
                raise TypeError
            self.possible_agents = np.arange(num_hps)

            self.hp_names = []
            if "config_space" in self.config:
                self.hp_names = self.config["config_space"].get_hyperparameter_names()
            self.max_num_agent = len(self.possible_agents)
            self.env_step = self.step
            self.env_reset = self.reset
            self.step = self.multi_agent_step
            self.reset = self.multi_agent_reset
            self.agents = []
            self.current_agent = None
            self.observation = None
            self.reward = None
            self.termination = False
            self.truncation = False
            self.info = None
            # TODO: this should be set to a reasonable default, ideally
            # Else playing with less than the full number of agents will be really hard
            if "default_action" in self.config:
                self.action = self.config.default_action
            else:
                self.action = self.action_space.sample()

            self.observation_spaces = {}
            for a in self.possible_agents:
                self.observation_spaces[a] = self.observation_space

            space_class = type(self.action_space)
            if space_class == gym.spaces.Box:
                lowers = self.action_space.low
                uppers = self.action_space.high
            elif space_class == gym.spaces.MultiDiscrete:
                num_options = list(self.action_space.nvec)
            self.action_spaces = {}
            for a in self.possible_agents:
                if space_class == gym.spaces.Box:
                    subspace = gym.spaces.Box(
                        low=np.array([lowers[a]]), high=np.array([uppers[a]])
                    )
                elif space_class == gym.spaces.Dict:
                    subspace = self.action_space.spaces[self.hp_names[a]]
                else:
                    subspace = gym.spaces.Discrete(num_options[a])
                self.action_spaces[a] = subspace

    def multi_agent_reset(self, seed: int | None = None):
        """Reset env, but don't return observations.

        Parameters
        ----------
        seed : int
            seed to use

        """
        self.observation, self.info = self.env_reset(seed)

    def last(self):
        """Get current step data.

        Returns:
        --------
        np.array, float, bool, bool, dict
        """
        return (
            self.observation,
            self.reward,
            self.termination,
            self.truncation,
            self.info,
        )

    def multi_agent_step(self, action):
        """Step for a single hyperparameter.

        Parameters
        ----------
        action
            the action in the current agent's dimension

        """
        if isinstance(action, dict):
            self.action[next(iter(action.keys()))] = next(iter(action.values()))
        elif isinstance(self.action, dict):
            self.action[list(self.action.keys())[self.current_agent]] = action
        else:
            self.action[self.current_agent] = action
        self.current_agent = self.agents.index(self.current_agent) + 1
        if self.current_agent >= len(self.agents):
            (
                self.observation,
                self.reward,
                self.termination,
                self.truncation,
                self.info,
            ) = self.env_step(self.action)
            self.current_agent = self.agents[0]

    def register_agent(self, agent_id):
        """Add agent.

        Parameters
        ----------
        agent_id : int
            id of the agent to add

        """
        if isinstance(agent_id, str):
            if len(agent_id) > 1:
                if agent_id in self.hp_names:
                    agent_id = self.hp_names.index(agent_id)
                else:
                    raise ValueError(
                        """Unknown agent name - please register an ID
                        or a name within the config space."""
                    )
            else:
                agent_id = int(agent_id)
        assert agent_id not in self.agents
        assert agent_id in self.possible_agents
        self.agents.append(agent_id)
        if self.current_agent is None:
            self.current_agent = agent_id

    def remove_agent(self, agent_id):
        """Remove agent.

        Parameters
        ----------
        agent_id : int
            id of the agent to remove

        """
        if agent_id in self.agents:
            self.agents.remove(agent_id)

    @property
    def num_agents(self):
        """Current number of agents."""
        return len(self.agents)

    @property
    def agent_selection(self):
        """Current agent."""
        return self.current_agent

    @property
    def infos(self):
        """Current infos per agent."""
        infos = {}
        for a in self.agents:
            infos[a] = self.info
        return infos

    @property
    def rewards(self):
        """Current rewards values per agent."""
        rewards = {}
        for a in self.agents:
            rewards[a] = self.rewards
        return rewards

    @property
    def terminations(self):
        """Current termination values per agent."""
        terminations = {}
        for a in self.agents:
            terminations[a] = self.termination
        return terminations

    @property
    def truncations(self):
        """Current truncation values per agent."""
        truncations = {}
        for a in self.agents:
            truncations[a] = self.truncation
        return truncations
