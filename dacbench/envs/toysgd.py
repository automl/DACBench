from dacbench import AbstractEnv
import numpy as np
from numpy.polynomial import Polynomial
from typing import Union, Tuple, Optional
import pandas as pd


def create_polynomial_instance_set(
        out_fname: str,
        n_samples: int = 100,
        order: int = 2,
        low: float = -10,
        high: float = 10
):
    instances = []
    for i in range(n_samples):
        coeffs = sample_coefficients(order=order, low=low, high=high)
        instance = {
            "ID": i,
            "family": "polynomial",
            "order": order,
            "low": low,
            "high": high,
            "coefficients": coeffs
        }
        instances.append(instance)
    df = pd.DataFrame(instances)
    df.to_csv(out_fname, sep=";", index=False)


def sample_coefficients(order: int = 2, low: float = -10, high: float = 10):
    n_coeffs = order + 1
    coeffs = np.zeros((n_coeffs,))
    coeffs[0] = np.random.uniform(0, high, size=1)
    coeffs[1:] = np.random.uniform(low, high, size=n_coeffs-1)
    return coeffs


class ToySGDEnv(AbstractEnv):
    """
    Optimize toy functions with SGD + Momentum.


    """
    def __init__(self, config):
        super(ToySGDEnv, self).__init__(config)
        self.n_steps_max = config.get("cutoff", 1000)

        self.velocity = 0
        self.gradient = 0
        self.history = []
        self.n_dim = None  # type: Optional[int]
        self.objective_function = None
        self.objective_function_deriv = None
        self.x_min = None
        self.f_min = None
        self.x_cur = None
        self.f_cur = None
        self.momentum = 0  # type: Optional[float]
        self.learning_rate = None  # type: Optional[float]
        self.n_steps = 0  # type: Optional[int]

    def build_objective_function(self):
        if self.instance["family"] == "polynomial":
            order = int(self.instance["order"])
            if order != 2:
                raise NotImplementedError("Only order 2 is currently implemented for polynomial functions.")
            self.n_dim = order
            coeffs_str = self.instance["coefficients"]
            coeffs_str = coeffs_str.strip("[]")
            coeffs = [float(item) for item in coeffs_str.split()]
            self.objective_function = Polynomial(coef=coeffs)
            self.objective_function_deriv = self.objective_function.deriv(m=1)  # lambda x0: derivative(self.objective_function, x0, dx=1.0, n=1, args=(), order=3)
            self.x_min = - coeffs[1] / (2 * coeffs[0] + 1e-10)  # add small epsilon to avoid numerical instabilities
            self.f_min = self.objective_function(self.x_min)

            self.x_cur = self.get_initial_position()
        else:
            raise NotImplementedError("No other function families than polynomial are currently supported.")

    def get_initial_position(self):
        return 0  # np.random.uniform(-5, 5, size=self.n_dim-1)

    def step(self, action: Union[float, Tuple[float, float]]):
        done = False
        info = {}

        # parse action
        if np.isscalar(action):
            log_momentum = 1
            log_learning_rate = action
        elif len(action) == 2:
            log_learning_rate, log_momentum = action
        else:
            raise ValueError
        self.momentum = 10 ** log_momentum
        self.learning_rate = 10 ** log_learning_rate

        # SGD + Momentum update
        self.velocity = self.momentum * self.velocity + self.learning_rate * self.gradient
        self.x_cur -= self.velocity
        self.gradient = self.objective_function_deriv(self.x_cur)

        # State
        remaining_budget = self.n_steps_max - self.n_steps
        state = {
            "remaining_budget": remaining_budget,
            "gradient": self.gradient,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum
        }

        # Reward
        # current function value
        self.f_cur = self.objective_function(self.x_cur)
        # log regret
        log_regret = np.log(np.abs(self.f_min - self.f_cur))
        reward = -log_regret

        self.history.append(self.x_cur)

        # Stop criterion
        self.n_steps += 1
        if self.n_steps > self.n_steps_max:
            done = True

        return state, reward, done, info

    def reset(self):
        """
        Reset environment

        Returns
        -------
        np.array
            Environment state
        """
        super(ToySGDEnv, self).reset_()

        self.velocity = 0
        self.gradient = 0
        self.history = []
        self.objective_function = None
        self.objective_function_deriv = None
        self.x_min = None
        self.f_min = None
        self.x_cur = None
        self.f_cur = None
        self.momentum = 0
        self.learning_rate = 0
        self.n_steps = 0
        self.build_objective_function()
        return {
            "remaining_budget": self.n_steps_max,
            "gradient": self.gradient,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum
        }

    def render(self, **kwargs):
        import matplotlib.pyplot as plt
        history = np.array(self.history).flatten()
        X = np.linspace(1.05 * np.amin(history), 1.05 * np.amax(history), 100)
        Y = self.objective_function(X)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(X, Y)
        ax.plot(history, self.objective_function(history), marker="x", color="black")
        ax.plot(self.x_cur, self.objective_function(self.x_cur), marker="x", color="red")
        plt.show()

    def close(self):
        pass





