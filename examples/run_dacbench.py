from dacbench.runner import run_dacbench, plot_results
from dacbench.agents import RandomAgent


def make_agent(env):
    return RandomAgent(env)


path = "dacbench_tabular"
run_dacbench(path, make_agent, 2)
plot_results(path)
