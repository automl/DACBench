from gym import Wrapper
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sb

sb.set_style("darkgrid")
current_palette = list(sb.color_palette())


class ActionFrequencyWrapper(Wrapper):
    def __init__(self, env, tracking_interval=None):
        super(ActionFrequencyWrapper, self).__init__(env)
        self.tracking_interval = tracking_interval
        self.overall = []
        if self.tracking_interval:
            self.interval_list = []
            self.current_interval = []
        self.action_space_type = type(self.env.action_space)

    def __setattr__(self, name, value):
        if name in [
            "tracking_interval",
            "overall",
            "interval_list",
            "current_interval",
            "env",
            "get_actions",
            "step",
            "render_action_tracking",
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
            "env",
            "get_actions",
            "step",
            "render_action_tracking",
        ]:
            return object.__getattribute__(self, name)
        else:
            return getattr(self.env, name)

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
        self.overall.append(action)
        if self.tracking_interval:
            if len(self.current_interval) < self.tracking_interval:
                self.current_interval.append(action)
            else:
                self.interval_list.append(self.current_interval)
                self.current_interval = [action]
        return state, reward, done, info

    def get_actions(self):
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
    def render_action_tracking(self):
        """
        Render action progression

        Returns
        -------
        np.array
            RBG data of action tracking

        """

        def plot_single(ax=None, index=None, title=None, x=False, y=False):
            if ax is None:
                plt.xlabel("Step")
                plt.ylabel("Action value")
            elif x and y:
                ax.set_ylabel("Action value")
                ax.set_xlabel("Step")
            elif x:
                ax.set_xlabel("Step")
            elif y:
                ax.set_ylabel("Action value")

            if index is not None:
                ys = [state[index] for state in self.overall]
            else:
                ys = self.overall

            if ax is None:
                p = plt.plot(
                    np.arange(len(self.overall)), ys, label="Step actions", color="g",
                    )
            else:
                p = ax.plot(
                    np.arange(len(self.overall)), ys, label="Step actions", color="g",
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
                        np.arange(len(self.interval_list))*self.tracking_interval,
                        [np.mean(interval) for interval in y_ints],
                        label="Mean interval action", color="orange"
                        )
                    plt.legend(loc="upper left")
                else:
                    p2 = ax.plot(
                        np.arange(len(self.interval_list))*self.tracking_interval,
                        [np.mean(interval) for interval in y_ints],
                        label="Mean interval action", color="orange"
                        )
                    ax.legend(loc="upper left")
            return p, p2

        if self.action_space_type == spaces.Discrete:
            figure = plt.figure(figsize=(12, 6))
            canvas = FigureCanvas(figure)
            p, p2 = plot_single()
            canvas.draw()
        elif self.action_space_type == spaces.Dict:
            raise NotImplementedError
        elif self.action_space_type == spaces.Tuple:
            raise NotImplementedError
        elif self.action_space_type == spaces.MultiDiscrete or self.action_space_type == spaces.MultiBinary or self.action_space_type == spaces.Box:
            if self.action_space_type == spaces.MultiDiscrete:
                action_size = len(self.env.observation_space.nvec)
            elif self.action_space_type == spaces.MultiBinary:
                action_size = self.env.observation_space.n
            else:
                action_size = len(self.env.observation_space.high)
            if action_size < 5:
                figure, axarr = plt.subplots(action_size)
            else:
                dim = action_size%4
                figure, axarr = plt.subplots((action_size%4)+1, action_size//dim)
            figure.suptitle("State over time")
            canvas = FigureCanvas(figure)
            for i in range(action_size):
                x=False
                if i%dim==dim-1:
                    x=True
                if action_size < 5:
                    p, p2 = plot_single(axarr[i], i,y=True, x=x)
                else:
                    y = i%action_size//dim==0
                    p, p2 = plot_single(axarr[i%dim, i//dim], i, x=x, y=y)
            for i in range(action_size//dim-1):
                figure.delaxes(axarr[dim, i])
            canvas.draw()
        width, height = figure.get_size_inches() * figure.get_dpi()
        img = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(
            int(height), int(width), 3
        )
        return img
