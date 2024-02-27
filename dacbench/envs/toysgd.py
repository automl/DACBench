"""Environment for sgd with toy functions."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from dacbench import AbstractMADACEnv

if TYPE_CHECKING:
    from numpy.random import Generator


def create_polynomial_instance_set(
    out_fname: str,
    rng: Generator,
    n_samples: int = 100,
    order: int = 2,
    low: float = -10,
    high: float = 10,
):
    """Make instance set."""
    instances = []
    for i in range(n_samples):
        coeffs = sample_coefficients(rng, order=order, low=low, high=high)
        instance = {
            "ID": i,
            "family": "polynomial",
            "order": order,
            "low": low,
            "high": high,
            "coefficients": coeffs,
        }
        instances.append(instance)
    instance_df = pd.DataFrame(instances)
    instance_df.to_csv(out_fname, sep=";", index=False)


def sample_coefficients(
    rng: Generator, order: int = 2, low: float = -10, high: float = 10
):
    """Sample function coefficients."""
    n_coeffs = order + 1
    coeffs = np.zeros((n_coeffs,))
    coeffs[0] = rng.uniform(0, high, size=1)
    coeffs[1:] = rng.uniform(low, high, size=n_coeffs - 1)
    return coeffs


def create_noisy_quadratic_instance_set(
    out_fname: str,
    rng: Generator,
    n_samples: int = 100,
    low: float = -10,
    high: float = 10,
):
    """Make instance set."""
    instances = []
    for i in range(n_samples):
        h = rng.uniform(0, high)
        c = rng.uniform(low, high)
        instance = {
            "ID": i,
            "family": "noisy_quadratic",
            "h": h,
            "c": c,
        }
        instances.append(instance)
    instance_df = pd.DataFrame(instances)
    instance_df.to_csv(out_fname, sep=";", index=False)
    return instance_df


class ToySGDEnv(AbstractMADACEnv):
    """Optimize toy functions with SGD + Momentum.

    Action: [log_learning_rate, log_momentum] (log base 10)
    State: Dict with entries remaining_budget, gradient, learning_rate, momentum
    Reward: negative log regret of current and true function value

    An instance can look as follows:
    ID                                                  0
    family                                     polynomial
    order                                               2
    low                                                -2
    high                                                2
    coefficients    [ 1.40501053 -0.59899755  1.43337392]

    """

    def __init__(self, config):
        """Init env."""
        super().__init__(config)

        if config["batch_size"]:
            self.batch_size = config["batch_size"]
        self.velocity = 0
        self.gradient = np.zeros(self.batch_size)
        self.history = []
        self.n_dim = None
        self.objective_function = None
        self.objective_function_deriv = None
        self.x_min = None
        self.f_min = None
        self.x_cur = None
        self.f_cur = None
        self.momentum = 0
        self.learning_rate = None
        self.rng = np.random.default_rng(self.initial_seed)

    def build_objective_function(self):
        """Make base function."""
        if self.instance["family"] == "polynomial":
            order = int(self.instance["order"])
            if order != 2:
                raise NotImplementedError(
                    "Only order 2 is currently implemented for polynomial functions."
                )
            self.n_dim = order
            coeffs_str = self.instance["coefficients"]
            coeffs_str = coeffs_str.strip("[]")
            coeffs = [float(item) for item in coeffs_str.split()]
            self.objective_function = Polynomial(coef=coeffs)
            self.objective_function_deriv = self.objective_function.deriv(m=1)
            self.x_min = -coeffs[1] / (
                2 * coeffs[0] + 1e-10
            )  # add small epsilon to avoid numerical instabilities
            self.f_min = self.objective_function(self.x_min)

            self.x_cur = self.get_initial_positions()
        elif self.instance["family"] == "noisy_quadratic":
            h = self.instance["h"]
            c = self.instance["c"]
            self.objective_function = lambda theta: 0.5 * np.sum(h * (theta - c) ** 2)
            self.objective_function_deriv = lambda theta: np.sum(h * (theta - c))
            self.x_min = np.sum(h * c) / np.sum(h)
            self.f_min = self.objective_function(self.x_min)
            self.x_cur = self.get_initial_positions()
        else:
            raise NotImplementedError(
                "No other function families than polynomial and "
                "noisy_quadratic are currently supported."
            )

    def get_initial_positions(self):
        """Get number of batch_size initial positions."""
        return self.rng.uniform(-5, 5, size=self.batch_size)

    def step(
        self, action: float | tuple[float, float]
    ) -> tuple[dict[str, float], float, bool, dict]:
        """Take one step with SGD.

        Parameters
        ----------
        action: Tuple[float, Tuple[float, float]]
            If scalar, action = (log_learning_rate)
            If tuple, action = (log_learning_rate, log_momentum)

        Returns:
        -------
        Tuple[Dict[str, float], float, bool, Dict]

            - state : Dict[str, float]
                State with entries:
                "remaining_budget", "gradient", "learning_rate", "momentum"
            - reward : float
            - terminated : bool
            - truncated : bool
            - info : Dict

        """
        truncated = super().step_()
        info = {}

        # parse action
        if np.isscalar(action):
            log_learning_rate = action
        elif len(action) == 2:
            log_learning_rate, log_momentum = action
            self.momentum = 10**log_momentum
        else:
            raise ValueError
        self.learning_rate = 10**log_learning_rate

        # SGD + Momentum update
        self.velocity = (
            self.momentum * self.velocity + self.learning_rate * self.gradient
        )
        self.x_cur -= self.velocity
        self.gradient = self.objective_function_deriv(self.x_cur)

        # State
        remaining_budget = self.n_steps - self.c_step
        state = {
            "remaining_budget": remaining_budget,
            "gradient": self.gradient,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
        }

        # Reward
        # current function value
        self.f_cur = self.objective_function(self.x_cur)
        # log regret
        log_regret = np.log10(np.abs(self.f_min - self.f_cur))
        reward = -np.mean(log_regret)

        self.history.append(self.x_cur)

        return state, reward, False, truncated, info

    def reset(self, seed=None, options=None):
        """Reset environment.

        Parameters
        ----------
        seed : int
            seed
        options : dict
            options dict (not used)

        Returns:
        -------
        np.array
            Environment state
        dict
            Meta-info

        """
        if options is None:
            options = {}
        super().reset_(seed)

        self.velocity = 0
        self.gradient = np.zeros(self.batch_size)
        self.history = []
        self.objective_function = None
        self.objective_function_deriv = None
        self.x_min = None
        self.f_min = None
        self.x_cur = None
        self.f_cur = None
        self.momentum = 0
        self.learning_rate = 0
        self.build_objective_function()
        remaining_budget = self.n_steps - self.c_step
        return {
            "remaining_budget": remaining_budget,
            "gradient": self.gradient,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
        }, {}

    def render(self, **kwargs):
        """Render progress."""
        import matplotlib.pyplot as plt

        history = np.array(self.history).flatten()
        X = np.linspace(1.05 * np.amin(history), 1.05 * np.amax(history), 100)
        Y = self.objective_function(X)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(X, Y, label="True")
        ax.plot(
            history,
            self.objective_function(history),
            marker="x",
            color="black",
            label="Observed",
        )
        ax.plot(
            self.x_cur,
            self.objective_function(self.x_cur),
            marker="x",
            color="red",
            label="Current Optimum",
        )
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("instance: " + str(self.instance["coefficients"]))
        plt.show()

    def close(self):
        """Close env."""
