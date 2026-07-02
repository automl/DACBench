"""AlphaRuleNet policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from ConfigSpace import ConfigurationSpace, Float
from torch import nn

from dacbench.envs.dacboenv.policy.abstract_policy import AbstractPolicy

if TYPE_CHECKING:
    from dacbench.envs.dacboenv.dacboenv import DACBOEnv
    from dacbench.envs.dacboenv.env.observations.types import ObsType


def get_nweights_alpharulenet() -> int:
    """Get the number of weights of AlphaRuleNet.

    Returns:
    -------
    int
        The number of weights.
    """
    return AlphaRuleNet.n_weights


# TODO add torch no grad
class AlphaRuleNet(nn.Module):
    """Alpha Rule Net.

    Inspired by an analytical form of SAWEI.
    """

    n_weights = 57

    def __init__(
        self,
        delta_alpha: float = 0.1,
        k: float = 10.0,
        weights: list[float] | None = None,
    ):
        """Initialize.

        Parameters
        ----------
        env : DACBOEnv
            The DACBO environment.
        delta_alpha : float, optional
            The amount of adjustment of alpha, same in SAWEI paper, by default 0.1
        k : float, optional
            How steep the comparison between the acq fun values of PI and EI should be, by default 10
        weights : list[float] | None, optional
            The weight vector, by default None. If None, initialize to the approximation of SAWEI rule.
        """
        super().__init__()
        self.delta_alpha = delta_alpha
        self.k = k

        # 5 inputs (R, v_PI, v_EI, alpha_prev, R_scale)
        self.fc1 = nn.Linear(5, 8)
        self.fc2 = nn.Linear(8, 1)

        if weights is None:
            # Ensures very rough SAWEI behavior
            # Default preinitialization (as before)
            nn.init.constant_(self.fc2.bias, 0.0)
            self.fc1.weight.data.zero_()
            self.fc1.bias.data.zero_()
            # Neurons 0-3 approximate tanh(k*(v_EI - v_PI))
            self.fc1.weight.data[0, 1] = -k
            self.fc1.weight.data[1, 2] = k
            self.fc1.weight.data[2, 1] = -0.5 * k
            self.fc1.weight.data[2, 2] = 0.5 * k
            self.fc1.weight.data[3, 1] = -0.5 * k
            self.fc1.weight.data[3, 2] = 0.5 * k
            # Neurons 4-7 approximate Gaussian gate using R / R_scale
            self.fc1.weight.data[4, 0] = -1.0
            self.fc1.weight.data[4, 4] = 1.0
            self.fc1.weight.data[5, 0] = 1.0
            self.fc1.weight.data[5, 4] = -1.0
            self.fc1.weight.data[6, 0] = -0.5
            self.fc1.weight.data[6, 4] = 0.5
            self.fc1.weight.data[7, 0] = 0.5
            self.fc1.weight.data[7, 4] = -0.5
            # Output weights
            self.fc2.weight.data.fill_(delta_alpha * 0.125)
        else:
            # Flatten all parameters into a single vector
            params = list(self.fc1.parameters()) + list(self.fc2.parameters())
            # Count total elements
            total_params = sum(p.numel() for p in params)
            assert (
                len(weights) == total_params
            ), f"Expected {total_params} floats, got {len(weights)}"
            # Copy values
            offset = 0
            for p in params:
                n = p.numel()
                p.data.copy_(
                    torch.tensor(weights[offset : offset + n], dtype=p.dtype).view_as(p)
                )
                offset += n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference.

        Parameters
        ----------
        x : torch.Tensor
            Input (obs: R, v_PI, v_EI, alpha_prev, R_scale)

        Returns:
        -------
        torch.Tensor
            Output (alpha).
        """
        x[:, 0:1]
        x[:, 1:2]
        x[:, 2:3]
        alpha_prev = x[:, 3:4]
        x[:, 4:5]

        # Hidden layer
        h = torch.tanh(self.fc1(x))

        # Linear output
        delta_alpha_out = self.fc2(h)

        # Update alpha
        alpha_new = alpha_prev + delta_alpha_out
        return torch.clamp(alpha_new, 0.0, 1.0)

    @classmethod
    def alpha_rule_init_weights(
        cls: type[AlphaRuleNet], k: float = 10, delta_alpha: float = 0.1
    ) -> torch.Tensor:
        """Construct the flattened parameter vector that reproduces the default
        (hand-crafted) initialization of AlphaRuleNet when `weights=None`.

        This function generates the exact 57-element weight vector corresponding
        to the analytical SAWEI-inspired initialization used in the model:
        - fc1.weight: 8x5 matrix encoding comparisons between PI, EI, and scaled R
        - fc1.bias: all zeros
        - fc2.weight: uniform weights equal to delta_alpha / 8
        - fc2.bias: zero

        The returned vector matches the internal PyTorch parameter ordering:
            [fc1.weight, fc1.bias, fc2.weight, fc2.bias]

        Parameters
        ----------
        k : float
            Steepness parameter controlling sensitivity to the difference
            between acquisition values (e.g., EI vs PI).
        delta_alpha : float
            Step size scaling factor applied to the output of the network.

        Returns:
        -------
        torch.Tensor
            A 1D tensor of length 57 containing the initialized network parameters.
            This tensor can be passed directly as the `weights` argument when
            constructing an `AlphaRuleNet`.
        """
        weights: list[float] = []

        # ---- fc1.weight (8 x 5) ----
        fc1_weight: list[list[float]] = [
            [0, -k, 0, 0, 0],
            [0, 0, k, 0, 0],
            [0, -0.5 * k, 0.5 * k, 0, 0],
            [0, -0.5 * k, 0.5 * k, 0, 0],
            [-1, 0, 0, 0, 1],
            [1, 0, 0, 0, -1],
            [-0.5, 0, 0, 0, 0.5],
            [0.5, 0, 0, 0, -0.5],
        ]

        for row in fc1_weight:
            weights.extend(row)

        # ---- fc1.bias (8) ----
        weights.extend([0.0] * 8)

        # ---- fc2.weight (1 x 8) ----
        weights.extend([delta_alpha * 0.125] * 8)

        # ---- fc2.bias (1) ----
        weights.append(0.0)

        return torch.tensor(weights, dtype=torch.float32)


class AlphaRulePolicy(AbstractPolicy):
    """AlphaRulePolicy.

    Expects the sawei observations of ubr_difference, acq_value_PI, acq_value_EI, previous_param.
    Interface to DACBOEnv.
    """

    def __init__(
        self,
        env: DACBOEnv,
        alpha_start: float = 0.5,
        delta_alpha: float = 0.1,
        k: float = 10,
        weights: list[float] | None = None,
    ) -> None:
        """Initialize.

        Parameters
        ----------
        env : DACBOEnv
            The DACBO environment.
        alpha_start : float, optional
            The start value of alpha, by default 0.5. This is basically EI.
        delta_alpha : float, optional
            The amount of adjustment of alpha, same in SAWEI paper, by default 0.1
        k : float, optional
            How steep the comparison between the acq fun values of PI and EI should be, by default 10
        weights : list[float] | None, optional
            The weight vector, by default None. If None, initialize to the approximation of SAWEI rule.
        """
        super().__init__(env)
        self.delta_alpha = delta_alpha
        self.alpha_start = alpha_start
        self.k = k
        self.weights = weights

        self.net = AlphaRuleNet(delta_alpha=delta_alpha, k=k, weights=self.weights)
        self._ubr_diffs: list[float] = []

    def __call__(self, obs: ObsType) -> int | float | list[float] | None:
        """Infer action based on observations.

        Calculate/track the scale of the difference of the UBR here.

        Parameters
        ----------
        obs : dict[str, Any]
            The observations.

        Returns:
        -------
        int | float | list[float] | None
            The action, the WEI alpha parameter.
        """
        self._ubr_diffs.append(obs["ubr_smoothed_gradient"])
        ubr_diff_std = np.std(self._ubr_diffs)
        if np.isnan(ubr_diff_std):
            ubr_diff_std = 1
        previous_param = (
            float(obs["previous_param"])
            if obs["previous_param"] is not None
            else self.alpha_start
        )
        x_list = [
            float(obs["ubr_smoothed_gradient"]),
            float(obs["acq_value_PI"]),
            float(obs["acq_value_EI"]),
            previous_param,
            float(ubr_diff_std),
        ]
        x = torch.tensor([x_list], dtype=torch.float32)  # batch
        y = torch.squeeze(self.net(x).detach().cpu())
        return float(y)

    def set_seed(self, seed: int | None) -> None:
        """Set seed for the model.

        Parameters
        ----------
        seed : int | None
            Seed
        """
        torch.manual_seed(seed)

    @classmethod
    def get_alpharulenet_configspace(
        cls, weight_bounds: tuple[float, float], k: float = 10, delta_alpha: float = 0.1
    ) -> ConfigurationSpace:
        """Get configuration space for AlphaRuleNet policy.

        Parameters
        ----------
        weight_bounds : tuple[float,float]
            The weight bounds.

        Returns:
        -------
        ConfigurationSpace
            The configuration space, contaings n_obs + 1 hyperparameters (weight vector and bias).
        """
        n_hps = AlphaRuleNet.n_weights
        defaults = AlphaRuleNet.alpha_rule_init_weights(k=k, delta_alpha=delta_alpha)
        configspace = ConfigurationSpace()
        configspace.add(
            [
                Float(name=f"w{i}", bounds=weight_bounds, default=float(defaults[i]))
                for i in range(n_hps)
            ]
        )
        return configspace
