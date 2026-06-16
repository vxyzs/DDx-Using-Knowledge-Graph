from abc import ABC, abstractmethod
from typing import List, Tuple, Any

class BaseSymptomRetriever(ABC):
    """
    Abstract base class defining the contract for symptom retriever components.
    Retrieves matching evidence nodes from the Knowledge Graph based on a text query.
    """

    @abstractmethod
    def retrieve(self, text: str) -> str:
        """
        Analyze raw user query and retrieve evidence descriptions as a JSON string.

        Args:
            text (str): Natural language description of symptoms.

        Returns:
            str: JSON serialized string of matched evidence node attributes.
        """
        pass

class BaseSymptomParser(ABC):
    """
    Abstract base class defining the contract for symptom parsing components.
    Uses contextual retrieved symptoms to generate structured evidence IDs and values.
    """

    @abstractmethod
    def parse_query(self, text: str, context: Any) -> Tuple[List[str], List[Any]]:
        """
        Parse raw patient dialogue under the provided retrieval context.

        Args:
            text (str): Dialogue or symptom query from patient/doctor.
            context (Any): Retreived evidence nodes context.

        Returns:
            Tuple[List[str], List[Any]]: A tuple of lists containing:
                - List of matched evidence IDs (e.g., 'E_123')
                - List of corresponding values ('YES', 'NO', or categorical/numerical value lists)
        """
        pass
