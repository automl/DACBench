from gym import spaces
from gym import Wrapper
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class StateTrackingWrapper(Wrapper):
    """ Wrapper to track state changed over time """

    def __init__(self, env, tracking_interval=None):
        super(StateTrackingWrapper, self).__init__(env)
        self.tracking_interval = tracking_interval
        self.overall = []
        if self.tracking_interval:
            self.interval_list = []
            self.current_interval = []
        self.episode = None
        self.state_type = type(env.observation_space)

    def __setattr__(self, name, value):
        if name in [
            "tracking_interval",
            "overall",
            "interval_list",
            "current_interval",
            "state_type",
            "env",
            "episode",
            "get_states",
            "step",
            "reset",
            "render_state_tracking",
        ]:
            object.__setattr__(self, name, value)
        else:
            setattr(self.env, name, value)

    def __getattribute__(self, name):
        if name in [
            "tracking_interval",
            "overall",
            "interval_list",
            "current_interval",
            "state_type",
            "env",
            "episode",
            "get_states",
            "step",
            "reset",
            "render_state_tracking",
        ]:
            return object.__getattribute__(self, name)
        else:
            return getattr(self.env, name)

    def reset(self):
        """
        Reset environment and record starting state

        Returns
        -------
        np.array
            state
        """
        state = self.env.reset()
        self.overall.append(state)
        if self.tracking_interval:
            if len(self.current_interval) < self.tracking_interval:
                self.current_interval.append(state)
            else:
                self.interval_list.append(self.current_interval)
                self.current_interval = [state]
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
        state, reward, done, info = self.env.step(action)
        self.overall.append(state)
        if self.tracking_interval:
            if len(self.current_interval) < self.tracking_interval:
                self.current_interval.append(state)
            else:
                self.interval_list.append(self.current_interval)
                self.current_interval = [state]
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
            complete_intervals = self.interval_list + [self.current_interval]
            return self.overall, complete_intervals
        else:
            return self.overall

    # TODO: test this
    def render_state_tracking(self):
        """
        Render state progression

        Returns
        -------
        np.array
            RBG data of state tracking

        """

        def plot_single(index=None, title=None):
            plt.title("State over time")
            plt.xlabel("Episode")
            plt.ylabel("State")
            if index:
                ys = [state[index] for state in self.overall]
            else:
                ys = self.overall
            p = plt.plot(
                np.arange(len(self.overall)), ys, label="Episode state", color="b"
            )
            p2 = None
            if self.tracking_interval:
                if index:
                    y_ints = []
                    for interval in self.interval_list:
                        y_ints.append([state[index] for state in interval])
                else:
                    y_ints = self.interval_list
                p2 = plt.plot(
                    np.arange(len(self.interval_list)),
                    [np.mean(interval) for interval in y_ints],
                    label="Interval state",
                    color="r",
                )

            return p, p2

        if self.state_type == spaces.Box:
            state_length = len(self.env.observation_space.high)
            # TODO: adjust max
            figure = plt.figure(figsize=(12, min(3 * state_length, 20)))
            canvas = FigureCanvas(figure)
            for i in range(state_length):
                p, p2 = plot_single(i)
                canvas.draw()
        elif self.state_type == spaces.Discrete:
            figure = plt.figure(figsize=(12, 6))
            canvas = FigureCanvas(figure)
            p, p2 = plot_single()
            canvas.draw()
        elif self.state_type == spaces.MultiDiscrete:
            state_length = len(self.env.observation_space.nvec)
            # TODO: adjust max
            figure = plt.figure(figsize=(12, min(3 * state_length, 20)))
            canvas = FigureCanvas(figure)
            for i in range(state_length):
                p, p2 = plot_single(i)
                canvas.draw()
        elif self.state_type == spaces.Dict:
            raise NotImplementedError
        elif self.state_type == spaces.Tuple:
            raise NotImplementedError
        elif self.state_type == spaces.MultiBinary:
            state_length = self.env.observation_space.n
            # TODO: adjust max
            figure = plt.figure(figsize=(12, min(3 * state_length, 20)))
            canvas = FigureCanvas(figure)
            for i in range(state_length):
                p, p2 = plot_single(i)
                canvas.draw()
        width, height = figure.get_size_inches() * figure.get_dpi()
        img = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(
            int(height), int(width), 3
        )
        return img
