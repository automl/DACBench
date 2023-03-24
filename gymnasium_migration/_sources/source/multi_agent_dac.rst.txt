.. _multi_agent_dac:

================
Multi-Agent DAC
================

.. role:: python(code)
    :language: python

As `Xue et al. <https://arxiv.org/abs/2210.06835>`_ have shown, multiple controllers collaborating to configure a single hyperparameter of the same algorithm each is a promising approach for solving DAC. 
To support further innovation in that direction, all of our environments with multiple configurable hyperparameters can be used as a Multi-Agent version.
This allows users to specify hyperparameters one by one instead of in a single step and thus is especially useful for those interfacing existing libraries.

In order to create a Multi-Agent DACBench environment, select either of the following benchmarks:

- :doc:`Sigmoid <benchmark_docs/sigmoid>` (Artificial Benchmark):
  Sigmoid function approximation in multiple dimensions.
- :doc:`ToySGD <benchmark_docs/toy_sgd>` (Artificial Benchmark):
  Controlling the learning rate in gradient descent.
- :doc:`Geometric <benchmark_docs/geometric>` (Artificial Benchmark):
  Approximating several functions at once.
- :doc:`ModCMA <benchmark_docs/modcma>`: Step-size & algorithm component control for EAs backed by IOHProfiler.

To instantiate a benchmark environment, first set the 'multi_agent' key in the configuration to True and then create the environment as usual:

.. code-block:: python

    from dacbench.benchmarks import SigmoidBenchmark
    bench = SigmoidBenchmark()
    bench.config["multi_agent"] = True
    env = bench.get_environment()

Running the benchmark is similar, but not quite the same as running a normal DACBench environment. First, you need to register the agents. 
Note that for this application, it makes sense to use an agent per hyperparameter even though it's technically possible to register less agents.
The remaining hyperparameters will be randomly, sampled, however, which could lead to adversarial effects.
To register an agent, use the ID of the hyperparameter you want to control. If using ConfigSpace, you can also use the hyperparameter's name:

.. code-block:: python

    from dacbench.agents import StaticAgent

    Agent_zero = StaticAgent(env, env.action_spaces[0].sample())
    Agent_one = StaticAgent(env, env.action_spaces[1].sample())
    agents = [Agent_zero, Agent_one]

    env.register_agent(0)
    env.register_agent(1)

The episode loop is slightly different as well:

.. code-block:: python

    env.reset()
    for agent in agents:
        observation, reward, terminated, truncated, info = env.last()
        action = agent.act(state, reward)
        env.step(action)

For more information on this interface, consult the `PettingZoo Documentation <https://pettingzoo.farama.org/content/basic_usage/>`_ on which our interface is based.

.. automodule:: dacbench.abstract_env
    :members:
    :show-inheritance: