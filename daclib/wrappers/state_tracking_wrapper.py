import gym
from gym import spaces
from gym import Wrapper


class StateTrackingWrapper(Wrapper):
    """ Wrapper to track state changed over time """

    def __init__(self, env, config):
        super(StateTrackingWrapper, self).__init__(env)
        tracking_interval = config["tracking_interval"]
        self.overall = []
        if self.tracking_interval:
            self.interval_list = []
            self.current_interval = []
        self.episode = None
        self.state_type = type(env.observation_space)

    def reset(self):
        """
        Reset environment and record starting state

        Returns
        -------
        np.array
            state
        """
        state = env.reset()
        self.episode = [state]
        return state

    def step(self, action):
        """
        Execute environment step and record state

        Parameters
        ----------
        action : int
            action to execute

        Returns
        -------
        np.array, float, bool, dict
            state, reward, done, metainfo
        """
        state, reward, done, info = env.step(action)
        self.episode.append(state)
        if done:
            if self.tracking_interval:
                if len(self.current_interval) < self.tracking_interval:
                    self.current_interval.append(self.episode)
                else:
                    self.interval_list.append(self.current_interval)
                    self.current_interval = [self.episode]
        return state, reward, done, info

    def get_states(self):
        """
        Get state progression

        Returns
        -------
        np.array or np.array, np.array
            all states or all states and interval sorted states

        """
        if self.tracking_interval:
            self.overall, self.interval_list
        else:
            self.overall

    # TODO: figure out rendering
    def render(self):
        if self.state_type == spaces.Box:
            pass
        elif self.state_type == spaces.Discrete:
            pass
        elif self.state_type == spaces.MultiDiscrete:
            pass
        elif self.state_type == spaces.Dict:
            pass
        elif self.state_type == spaces.Tuple:
            pass
        elif self.state_type == spaces.MultiBinary:
            pass
        return
