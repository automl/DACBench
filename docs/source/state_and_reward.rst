=============================
Modifying Observations & Cost
=============================

.. role:: python(code)
    :language: python

While all benchmarks in DACBench come with a default option for both the reward function and what observations about the algorithm are shown to the DAC controller,
both can be configured individually if needed.
The standard way of doing this on most benchmark is to use the config to provide a function. A very simple example could look like this:

.. code-block:: python

    from dacbench.benchmarks import SigmoidBenchmark

    def new_reward(env):
        if env.c_step % 2 == 0:
            return 1
        else:
            return 0

    bench = SigmoidBenchmark()
    bench.config.reward_function = new_reward
    modified_benchmark = bench.get_environment()

The environment itself is provided as an argument, so all internal information can be used to get the reward.
The same goes for the observations:

.. code-block:: python

    from dacbench.benchmarks import SigmoidBenchmark

    def new_obs(env):
        return env.remaining_budget

    bench = SigmoidBenchmark()
    bench.config.state_method = new_obs
    modified_benchmark = bench.get_environment()

If the config is logged, information about the updated functions is saved too, but for reusing this config, the code needs to be provided.
That means anyone that want to run a setting with altered rewards and observations needs the config plus the code of the new functions.
Therefore we advise to provide a file with only these functions in addition to the config - or make a PR to DACBench!