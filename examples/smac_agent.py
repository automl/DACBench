"""This code is adapted from the repo of the 'revisiting LR control' paper: https://github.com/automl/Revisiting_LR_Control/tree/main"""
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Integer
from dacbench.abstract_agent import AbstractDACBenchAgent
from dacbench.benchmarks import LubyBenchmark
from smac import (
    HyperparameterOptimizationFacade as HPOFacade,
    Scenario,
)
from smac.runhistory.dataclasses import TrialValue


class SMACAgent(AbstractDACBenchAgent):
    r"""Agent for Learning Rate Training in DACBench using SMAC
        This is a hacky solution to train SMAC using the ask-and-tell interface with episode rewards.

    Args:
        - env: DACBench environment
        - configspace: ConfigurationSpace for hyperparameters to optimize
        - n_trials: number of trials to perform by SMAC
    """

    def __init__(self, env, configspace, n_trials, hp_name="lr"):
        """Initialize the Agent."""
        self.scenario = Scenario(configspace, deterministic=True, n_trials=n_trials)

        intensifier = HPOFacade.get_intensifier(
            self.scenario,
            max_config_calls=1,  # We basically use one seed per config only
        )

        def dummy_train(self, config: Configuration, seed: int = 0) -> float:
            pass

        self.smac = HPOFacade(
            self.scenario,
            dummy_train,  # We pass the target function here
            intensifier=intensifier,
            overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
        )

        self.is_episode_beginning = True
        self.hp_name = hp_name

        super().__init__(env)

    def act(self, state=None, reward=None):
        """Returns the next action."""
        if self.is_episode_beginning:
            self.current_info = self.smac.ask()
            assert self.current_info.seed is not None
            self.is_episode_beginning = False

        return self.current_info.config[self.hp_name]

    def train(self, state=None, reward=None):  # noqa: D102
        pass

    def end_episode(self, state=None, reward=None):  # noqa: D102
        value = TrialValue(cost=-reward, time=0.5)
        self.smac.tell(self.current_info, value=value)
        self.is_episode_beginning = True


def run_smac():
    # Make environment
    bench = LubyBenchmark()
    env = bench.get_environment()

    # Set configuration space
    cs = ConfigurationSpace(seed=0)
    se = Integer("sequence_element", (0, np.log2(bench.config.cutoff)), default=0)
    cs.add(se)

    # Initialize agent
    agent = SMACAgent(env, configspace=cs, n_trials=10, hp_name="sequence_element")
    print("Here")

    # Execute 5 episodes
    rewards = []
    for _ in range(5):
        state, _ = env.reset()
        terminated, truncated = False, False
        reward = 0
        while not (terminated or truncated):
            # On the first step, 'ask' is called
            # Then we repeat the same action until the episode ends
            action = agent.act()
            _, reward, terminated, truncated, _ = env.step(action)

        # This is where 'tell' is called
        agent.end_episode(state, reward)
        rewards.append(reward)

    print(f"Final performance: {np.mean(rewards)} reward points.")


if __name__ == "__main__":
    run_smac()
