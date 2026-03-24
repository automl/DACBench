"""OmegaConf resolver helpers for DACBO task configuration."""

from __future__ import annotations

from ConfigSpace import ConfigurationSpace, Float


def get_dacbo_task_name(
    cost_type: str,
    action_space_id: str,
    observation_space_id: str,
    reward_id: str,
    instance_set_id: str,
) -> str:
    """Get task name for DACBO task.

    Parameters
    ----------
    cost_type : str
        The cost id of the target function.
    action_space_id : str
        The action space id of DACBOEnv.
    observation_space_id: str
        The observation space id of DACBOEnv.
    reward_id : str
        The reward id of DACBOEnv.
    instance_set_id : str
        The instance set id of DACBOEnv.

    Returns:
    -------
    str
        DACBO task name.
    """
    return f"dacbo_C{cost_type}_A{action_space_id}_S{observation_space_id}_R{reward_id}_I{instance_set_id}"


def get_perceptron_configspace(
    n_obs: int, weight_bounds: tuple[float, float]
) -> ConfigurationSpace:
    """Get configuration space for perceptron policy.

    Parameters
    ----------
    n_obs : int
        Number of observations.
    weight_bounds : tuple[float,float]
        The weight bounds.

    Returns:
    -------
    ConfigurationSpace
        The configuration space, contaings n_obs + 1 hyperparameters (weight vector and bias).
    """
    n_hps = n_obs + 1  # theta + bias
    configspace = ConfigurationSpace()
    configspace.add([Float(name=f"w{i}", bounds=weight_bounds) for i in range(n_hps)])
    return configspace
