""" Default configuration for trajectory optimization. """

TRAJ_OPT = {
    # Dual variable updates for non-PD Q-function.
    'del0': 1e-4,
    'min_eta': 1e-4,
    'max_eta': 1e16,
}
