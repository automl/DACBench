""" Default configuration and hyperparameter values for policies. """
INIT_LG = {
    'init_var': 1.0,
    'verbose': False
}

# PolicyPriorGMM
POLICY_PRIOR_GMM = {
    'min_samples_per_cluster': 20,
    'max_clusters': 50,
    'max_samples': 20,
    'strength': 1.0,
}
