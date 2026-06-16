import unittest

from core.interfaces import BaseSymptomParser, BaseSymptomRetriever


class TestInterfaces(unittest.TestCase):
    """
    Verify abstract base class interfaces correctness and instantiation rules.
    """

    def test_retriever_interface_enforces_abstract_method(self):
        class BadRetriever(BaseSymptomRetriever):
            pass

        with self.assertRaises(TypeError):
            BadRetriever()

    def test_parser_interface_enforces_abstract_method(self):
        class BadParser(BaseSymptomParser):
            pass

        with self.assertRaises(TypeError):
            BadParser()

    def test_valid_retriever_subclass(self):
        class GoodRetriever(BaseSymptomRetriever):
            def retrieve(self, text: str) -> str:
                return "{}"

        retriever = GoodRetriever()
        self.assertEqual(retriever.retrieve("fever"), "{}")

    def test_valid_parser_subclass(self):
        class GoodParser(BaseSymptomParser):
            def parse_query(self, text, context):
                return ["E_201"], ["YES"]

        parser = GoodParser()
        ev, val = parser.parse_query("test", "context")
        self.assertEqual(ev, ["E_201"])
        self.assertEqual(val, ["YES"])


if __name__ == "__main__":
    unittest.main()
