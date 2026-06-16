import json
import sys
import unittest
from unittest.mock import MagicMock, patch

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

from core.parser import Parser, PatientEvidences


class TestParser(unittest.TestCase):
    """
    Test suite for LLM-based symptom Parser.
    """

    @patch("core.parser.ChatOpenAI")
    @patch("core.parser.open")
    def setUp(self, mock_open, mock_chat_openai_cls):
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.read.return_value = "{}"
        mock_open.return_value = mock_file

        self.mock_llm = MagicMock()
        mock_chat_openai_cls.return_value = self.mock_llm

        self.parser = Parser()

    def test_parse_query_successful(self):
        # Set invoke mock to return a valid PatientEvidences model instance
        self.parser.output_parser.invoke.return_value = PatientEvidences(
            evidences=["E_1", "E_2"],
            values=["YES", ["E_2_@_V_1"]]
        )

        evidences, values = self.parser.parse_query("dialogue", "context")
        self.assertEqual(evidences, ["E_1", "E_2"])
        self.assertEqual(values, ["YES", ["E_2_@_V_1"]])

    def test_parser_exception_handling(self):
        self.mock_llm.invoke.side_effect = RuntimeError("API error")
        evidences, values = self.parser.parse_query("dialogue", "context")
        self.assertEqual(evidences, [])
        self.assertEqual(values, [])


if __name__ == "__main__":
    unittest.main()
