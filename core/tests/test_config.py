import unittest
from unittest.mock import MagicMock, patch

from core.config import load_config


class TestConfigLoader(unittest.TestCase):
    """
    Test suite for config.py configuration loader.
    """

    @patch("core.config.os.path.exists")
    def test_load_config_defaults_when_file_missing(self, mock_exists):
        mock_exists.return_value = False
        config = load_config()
        self.assertEqual(
            config["nlu"]["embedding_model"],
            "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        )
        self.assertEqual(config["traversal"]["smooth"], 1e-6)

    @patch("core.config.os.path.exists")
    @patch("core.config.open")
    def test_load_config_parses_json_correctly(self, mock_open, mock_exists):
        mock_exists.return_value = True
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.read.return_value = '{"nlu": {"thresh_evidence": 0.88}}'
        mock_open.return_value = mock_file

        config = load_config()
        self.assertEqual(config["nlu"]["thresh_evidence"], 0.88)
        self.assertEqual(config["traversal"]["smooth"], 1e-6)

    @patch("core.config.os.path.exists")
    @patch("core.config.open")
    def test_load_config_falls_back_on_corrupt_json(
        self, mock_open, mock_exists
    ):
        mock_exists.return_value = True
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.read.return_value = "{invalid json"
        mock_open.return_value = mock_file

        config = load_config()
        self.assertEqual(config["nlu"]["thresh_evidence"], 0.40)


if __name__ == "__main__":
    unittest.main()
