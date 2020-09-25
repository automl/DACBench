import gym


class AbstractEnv(gym.Env):
    """
    Abstract template for environments
    """

    def __init__(self, config):
        super(AbstractEnv, self).__init__()
        self.instance_set = config["instance_set"]
        self.inst_id = 0
        self.instance = self.instance_set[self.inst_id]

        self.n_steps = config["cutoff"]
        self.c_step = 0

        self.reward_range = config["reward_range"]
        self.observation_space = getattr(
            gym.spaces, config["observation_space"]
        )(
            *config["observation_space_args"],
            dtype=config["observation_space_type"]
        )
        self.action_space = getattr(gym.spaces, config["action_space"])(
            *config["action_space_args"]
        )

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

    def reset_(self):
        """
        Pre-reset function for round robin schedule through instance set
        """
        self.inst_id = (self.inst_id + 1) % len(self.instance_set)
        self.instance = self.instance_set[self.inst_id]
        self.c_step = 0

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_inst_id(self):
        """
        Return isntance ID

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

    def set_instance_set(self, inst_set):
        """
        Change instance set

        Parameters
        ----------
        inst_set: list
            New instance set
        """
        self.instance_set = inst_set

    def set_instance(self, instance):
        """
        Change currently used instance

        Parameters
        ----------
        instance:
            New instance
        """
        self.instance = instance
