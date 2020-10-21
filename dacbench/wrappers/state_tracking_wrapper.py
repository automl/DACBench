from gym import spaces
from gym import Wrapper
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sb

sb.set_style("darkgrid")
current_palette = list(sb.color_palette())


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

    def render_state_tracking(self):
        """
        Render state progression

        Returns
        -------
        np.array
            RBG data of state tracking

        """

        def plot_single(ax=None, index=None, title=None, x=False, y=False):
            if ax is None:
                plt.xlabel("Episode")
                plt.ylabel("State")
            elif x and y:
                ax.set_ylabel("State")
                ax.set_xlabel("Episode")
            elif x:
                ax.set_xlabel("Episode")
            elif y:
                ax.set_ylabel("State")

            if index is not None:
                ys = [state[index] for state in self.overall]
            else:
                ys = self.overall

            if ax is None:
                p = plt.plot(
                    np.arange(len(self.overall)), ys, label="Episode state", color="g",
                    )
            else:
                p = ax.plot(
                    np.arange(len(self.overall)), ys, label="Episode state", color="g",
                    )
            p2 = None
            if self.tracking_interval:
                if index is not None:
                    y_ints = []
                    for interval in self.interval_list:
                        y_ints.append([state[index] for state in interval])
                else:
                    y_ints = self.interval_list
                if ax is None:
                    p2 = plt.plot(
                        np.arange(len(self.interval_list)),
                        [np.mean(interval) for interval in y_ints],
                        label="Mean interval state", color="orange"
                        )
                    plt.legend(loc="upper left")
                else:
                    p2 = ax.plot(
                        np.arange(len(self.interval_list))*self.tracking_interval,
                        [np.mean(interval) for interval in y_ints],
                        label="Mean interval state", color="orange"
                        )
                    ax.legend(loc="upper left")
            return p, p2


        if self.state_type == spaces.Discrete:
            figure = plt.figure(figsize=(20, 20))
            canvas = FigureCanvas(figure)
            p, p2 = plot_single()
            canvas.draw()
        elif self.state_type == spaces.Dict:
            raise NotImplementedError
        elif self.state_type == spaces.Tuple:
            raise NotImplementedError
        elif self.state_type == spaces.MultiDiscrete or self.state_type == spaces.MultiBinary or self.state_type == spaces.Box:
            if self.state_type == spaces.MultiDiscrete:
                state_length = len(self.env.observation_space.nvec)
            elif self.state_type == spaces.MultiBinary:
                state_length = self.env.observation_space.n
            else:
                state_length = len(self.env.observation_space.high)
            if state_length < 5:
                dim = 1
                figure, axarr = plt.subplots(state_length)
            else:
                dim = state_length%4
                figure, axarr = plt.subplots(state_length%4, state_length//dim)
            figure.suptitle("State over time")
            canvas = FigureCanvas(figure)
            for i in range(state_length):
                x=False
                if i%dim==dim-1:
                    x=True
                if state_length < 5:
                    p, p2 = plot_single(axarr[i], i,y=True, x=x)
                else:
                    y = i%state_length//dim==0
                    p, p2 = plot_single(axarr[i%dim, i//dim], i, x=x, y=y)
            canvas.draw()
        width, height = figure.get_size_inches() * figure.get_dpi()
        img = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(
            int(height), int(width), 3
        )
        return img
