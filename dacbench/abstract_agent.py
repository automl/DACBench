class AbstractDACBenchAgent:
    """Abstract class to implement for use with the runner function."""

    def __init__(self, env):
        """
        Initialize agent.

        Parameters
        ----------
        env : gym.Env
            Environment to train on

        """
        pass

    def act(self, state, reward):
        """
        Compute and return environment action.

        Parameters
        ----------
        state
            Environment state
        reward
            Environment reward

        Returns
        -------
        action
            Action to take

        """
        raise NotImplementedError

    def train(self, next_state, reward):
        """
        Train during episode if needed (pass if not).

        Parameters
        ----------
        next_state
            Environment state after step
        reward
            Environment reward

        """
        raise NotImplementedError

    def end_episode(self, state, reward):
        """
        End of episode training if needed (pass if not).

        Parameters
        ----------
        state
            Environment state
        reward
            Environment reward

        """
        raise NotImplementedError
