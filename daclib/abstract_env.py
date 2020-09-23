import gym

class AbstractEnv(gym.Env):
    def __init__(self, config):
        self.super.__init__()
        self.instance_set = config['instance_set']
        self.inst_id = 0
        self.instance = self.instance_set[self.inst_id]

        self.n_steps = config['cuttoff']
        self.c_step = 0

        self.reward_range = config["reward_range"]
        self.observation_space = getattr(gym.spaces, config["observation_space"])(*config["observation_space_args"], dtype=config["observation_space_type"])
        self.action_space = getattr(gym.spaces, config["action_space"])(*config["action_space_args"])

    def step(self):
        done = False
        self.c_step += 1
        if self.c_step >= self.n_steps:
            done = True
        return done

    def reset(self):
        self.inst_id = (self.inst_id + 1) % len(self.instance_set)
        self.instance = self.instance_set[self.inst_id]
        self.c_step = 0

    def get_inst_id(self):
        return self.inst_id

    def get_instance_set(self):
        return self.instance_set

    def get_instance(self):
        return self.instance

    def set_inst_id(self, inst_id):
        self.inst_id = inst_id

    def set_instance_set(self):
        return self.instance_set

    def set_instance(self):
        return self.instance
