import json
import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock heavy modules before importing core.nlu
m_sentence = MagicMock()
sys.modules["sentence_transformers"] = m_sentence

m_sklearn = MagicMock()
sys.modules["sklearn"] = m_sklearn
sys.modules["sklearn.metrics.pairwise"] = m_sklearn.metrics.pairwise

import networkx as nx
import numpy as np

from core.nlu import DDxGraphNLU


class TestDDxGraphNLU(unittest.TestCase):
    """
    Test suite for DDxGraphNLU symptom retriever.
    """

    @patch("core.nlu.SentenceTransformer")
    @patch("core.nlu.open")
    def setUp(self, mock_open, mock_transformer_cls):
        self.mock_evidences = {
            "E_1": {"name": "fever", "data_type": "B"},
            "E_2": {"name": "cough", "data_type": "B"},
            "E_3": {"name": "eye_watering", "data_type": "B"},
        }
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.read.return_value = json.dumps(self.mock_evidences)
        mock_open.return_value = mock_file

        self.mock_transformer = MagicMock()
        mock_transformer_cls.return_value = self.mock_transformer
        self.mock_transformer.encode.side_effect = lambda texts: np.ones(
            (len(texts), 384)
        )

        self.G = nx.DiGraph()
        self.G.add_node(
            "E_1",
            type="evidence",
            data_type="B",
            question_en="fever",
            embedding=np.zeros(384).tolist(),
        )
        self.G.add_node(
            "E_2",
            type="evidence",
            data_type="B",
            question_en="cough",
            embedding=np.ones(384).tolist(),
        )
        self.G.add_node(
            "E_3",
            type="evidence",
            data_type="B",
            question_en="eye watering",
            embedding=(np.ones(384) * 0.5).tolist(),
        )

        self.nlu = DDxGraphNLU(self.G)

    def test_chunking_and_delimiters(self):
        query = "headache, fever; also chills but no cough"
        with patch.object(self.nlu, "_find_best_match") as mock_find:
            mock_find.return_value = [
                {"eid": "E_1", "value": [], "score": 0.8}
            ]
            self.nlu.parse_query1(query)
            called_chunks = [args[0] for args, _ in mock_find.call_args_list]
            self.assertIn("headache", called_chunks)
            self.assertIn("fever", called_chunks)
            self.assertIn("chills", called_chunks)

    def test_decimal_regex_protection(self):
        query = "temperature is 38.5 degrees"
        with patch.object(self.nlu, "_find_best_match") as mock_find:
            mock_find.return_value = []
            self.nlu.parse_query1(query)
            called_chunks = [args[0] for args, _ in mock_find.call_args_list]
            self.assertIn("temperature is 38.5 degrees", called_chunks)

    def test_negation_detection(self):
        query = "no fever"
        with patch.object(self.nlu, "_find_best_match") as mock_find:
            mock_find.return_value = [
                {"eid": "E_1", "value": [], "score": 0.9}
            ]
            all_findings, _, _ = self.nlu.parse_query1(query)
            self.assertEqual(len(all_findings), 1)
            self.assertTrue(all_findings[0]["negated"])

    @patch("core.nlu.cosine_similarity")
    def test_find_best_match_relative_filtering(self, mock_cosine):
        mock_cosine.return_value = np.array([[0.9, 0.4, 0.75]])
        matches = self.nlu._find_best_match("fever")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["eid"], "E_1")


if __name__ == "__main__":
    unittest.main()
