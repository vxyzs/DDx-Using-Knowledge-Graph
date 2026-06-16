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
        # Speed up unit tests by eliminating delays
        self.parser.retry_delay = 0.0

    def test_parse_query_successful(self):
        # Set invoke mock to return a valid PatientEvidences model instance
        self.parser.output_parser.invoke.return_value = PatientEvidences(
            evidences=["E_1", "E_2"],
            values=["YES", ["E_2_@_V_1"]]
        )

        evidences, values = self.parser.parse_query("dialogue", "context")
        self.assertEqual(evidences, ["E_1", "E_2"])
        self.assertEqual(values, ["YES", ["E_2_@_V_1"]])

    def test_parser_exception_handling_with_retry(self):
        # Primary model fails twice, then succeeds on the 3rd attempt
        self.parser.max_retries = 3
        mock_resp = MagicMock()
        mock_resp.content = "mock output"
        self.mock_llm.invoke.side_effect = [
            RuntimeError("API timeout 1"),
            RuntimeError("API timeout 2"),
            mock_resp
        ]
        self.parser.output_parser.invoke.return_value = PatientEvidences(
            evidences=["E_1"],
            values=["YES"]
        )

        evidences, values = self.parser.parse_query("dialogue", "context")
        self.assertEqual(self.mock_llm.invoke.call_count, 3)
        self.assertEqual(evidences, ["E_1"])
        self.assertEqual(values, ["YES"])

    @patch("core.parser.Parser._create_llm")
    def test_parser_fallback_mechanism(self, mock_create_llm):
        # Primary model fails completely, fallback model succeeds
        self.parser.max_retries = 2
        self.parser.fallback_models = ["backup-model"]

        # Mock primary model to fail twice
        self.mock_llm.invoke.side_effect = RuntimeError("Primary failed")

        # Mock fallback model to succeed
        mock_fallback_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "fallback output"
        mock_fallback_llm.invoke.return_value = mock_resp
        mock_create_llm.return_value = mock_fallback_llm

        self.parser.output_parser.invoke.return_value = PatientEvidences(
            evidences=["E_2"],
            values=["NO"]
        )

        evidences, values = self.parser.parse_query("dialogue", "context")
        self.assertEqual(self.mock_llm.invoke.call_count, 2)
        mock_create_llm.assert_called_once_with("backup-model")
        mock_fallback_llm.invoke.assert_called_once()
        self.assertEqual(evidences, ["E_2"])
        self.assertEqual(values, ["NO"])

    @patch("core.parser.Parser._create_llm")
    def test_parser_complete_failure(self, mock_create_llm):
        # Both primary and fallback models fail completely
        self.parser.max_retries = 2
        self.parser.fallback_models = ["backup-model"]

        # Mock primary model to fail twice
        self.mock_llm.invoke.side_effect = RuntimeError("Primary failed")

        # Mock fallback model to fail twice
        mock_fallback_llm = MagicMock()
        mock_fallback_llm.invoke.side_effect = RuntimeError("Fallback failed")
        mock_create_llm.return_value = mock_fallback_llm

        evidences, values = self.parser.parse_query("dialogue", "context")
        self.assertEqual(evidences, [])
        self.assertEqual(values, [])


if __name__ == "__main__":
    unittest.main()
