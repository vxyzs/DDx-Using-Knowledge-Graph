import math
import sys
import unittest
from unittest.mock import MagicMock

# Mock langchain and other missing third-party modules in sys.modules
m_openai = MagicMock()
sys.modules["langchain_openai"] = m_openai

m_core = MagicMock()
sys.modules["langchain_core"] = m_core
sys.modules["langchain_core.prompts"] = m_core.prompts
sys.modules["langchain_core.output_parsers"] = m_core.output_parsers

# Also mock sentence_transformers and sklearn
m_sentence = MagicMock()
sys.modules["sentence_transformers"] = m_sentence

m_sklearn = MagicMock()
sys.modules["sklearn"] = m_sklearn
sys.modules["sklearn.metrics.pairwise"] = m_sklearn.metrics.pairwise

import networkx as nx

from core.traversal import BaseTraversal


class MockTraversal(BaseTraversal):
    """
    Concrete implementation of BaseTraversal for testing.
    """
    def run(self, *args, **kwargs):
        pass


class TestTraversalMathAndLogic(unittest.TestCase):
    """
    Test suite for traversal math formulas and information gain algorithms.
    """

    def setUp(self):
        self.G = nx.DiGraph()
        
        self.G.add_node("C_1", type="condition")
        self.G.add_node("C_2", type="condition")
        
        self.G.add_node("E_1", type="evidence")
        self.G.add_node("E_2", type="evidence")
        
        self.G.add_edge("C_1", "E_1", p_e_given_c=0.9)
        self.G.add_edge("C_2", "E_1", p_e_given_c=0.1)
        self.G.add_edge("C_1", "E_2", p_e_given_c=0.5)
        self.G.add_edge("C_2", "E_2", p_e_given_c=0.5)

        self.traversal = MockTraversal(self.G)

    def test_safe_log(self):
        self.assertAlmostEqual(self.traversal.safe_log(0.0), math.log(1e-6))
        self.assertAlmostEqual(self.traversal.safe_log(0.5), math.log(0.5))

    def test_capped_add(self):
        self.assertEqual(self.traversal.capped_add(0.0, 1.0), 1.0)
        self.assertEqual(self.traversal.capped_add(0.0, 5.0), 2.0)
        self.assertEqual(self.traversal.capped_add(0.0, -3.0), -2.0)

    def test_convert_scores_to_probabilities(self):
        scores = {"C_1": 1.0, "C_2": 2.0}
        self.traversal._convert_scores_to_probabilities(scores)
        sum_p = sum(scores.values())
        self.assertAlmostEqual(sum_p, 1.0)
        self.assertTrue(scores["C_2"] > scores["C_1"])

    def test_compute_discriminating_evidence(self):
        scores = {"C_1": 0.0, "C_2": 0.0}
        asked = set()
        best_evidence = self.traversal._compute_discriminating_evidence(
            ["C_1", "C_2"], scores, asked
        )
        self.assertEqual(best_evidence, "E_1")

    def test_compute_discriminating_evidence_excludes_asked(self):
        scores = {"C_1": 0.0, "C_2": 0.0}
        asked = {"E_1"}
        best_evidence = self.traversal._compute_discriminating_evidence(
            ["C_1", "C_2"], scores, asked
        )
        self.assertEqual(best_evidence, "E_2")


if __name__ == "__main__":
    unittest.main()
