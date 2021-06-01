import unittest

from dacbench.agents import GenericAgent


class TestGenericAgent(unittest.TestCase):
    def test_dummy_policy(self):
        agent = GenericAgent(None, lambda env, state: 0)

        self.assertIsNone(agent.env)
        self.assertEqual(agent.act(None, None), 0)
        self.assertIsNone(agent.train(None, None))
        self.assertIsNone(agent.end_episode(None, None))
