from dacbench.runner import run_dacbench
from dacbench.agents import RandomAgent


# Function to create an agent fulfilling the DACBench Agent interface
# In this case: a simple random agent
def make_agent(env):
    return RandomAgent(env)


# Result output path
path = "dacbench_tabular"

# Run all DACBench benchmarks with the agent for 2 episodes each
run_dacbench(path, make_agent, 2)
