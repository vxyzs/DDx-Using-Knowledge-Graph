# Design Document: Core Unit Testing Suite

The testing suite under [core/tests/](../../core/tests/) provides comprehensive unit coverage for the DDx core logical modules.

---

## 1. Testing Philosophy & Mocking Strategy

To keep tests extremely fast (running in milliseconds) and independent of external API servers or massive machine learning models, the suite mocks all heavy dependencies using Python's built-in `sys.modules` cache and `unittest.mock` package.

### A. sys.path and sys.modules Injection
Because the Python test runner may be launched globally or from different working directories, dependencies like `langchain_core` and `sentence_transformers` might be missing or slow to load. We inject mock wrappers directly into the Python imports system:

```python
import sys
from unittest.mock import MagicMock

# Mock langchain and other missing third-party modules in sys.modules
m_openai = MagicMock()
sys.modules["langchain_openai"] = m_openai

m_core = MagicMock()
sys.modules["langchain_core"] = m_core
sys.modules["langchain_core.prompts"] = m_core.prompts
sys.modules["langchain_core.output_parsers"] = m_core.output_parsers
```

- **Why?**: This prevents `ModuleNotFoundError` during imports and intercepts requests to initialize active LLMs or download large embeddings, keeping tests fully offline and self-contained.

---

## 2. Test File Registry

### A. [test_interfaces.py](../../core/tests/test_interfaces.py)
- **Goal**: Verifies that standard abstract interfaces (`BaseSymptomRetriever` and `BaseSymptomParser`) correctly enforce instantiation rules.
- **Checks**:
  - Class instantiation raises `TypeError` if abstract methods are not defined.
  - Correct implementation subclasses execute and return successfully.

### B. [test_nlu.py](../../core/tests/test_nlu.py)
- **Goal**: Verifies natural language symptom chunking and similarity mapping.
- **Checks**:
  - Delimiter parsing splits complex sentences into separate symptom tokens.
  - Regex guards protect float values (e.g. "38.5") from being split.
  - Negation words ("no ", "not ") flag matches as negated.
  - Relative filtering removes matches scoring lower than `0.1` below the maximum.

### C. [test_parser.py](../../core/tests/test_parser.py)
- **Goal**: Verifies Pydantic parsing and output generation.
- **Checks**:
  - Map outputs into Pydantic models with custom lists.
  - Handle LangChain invoke responses.
  - Gracefully catch API exceptions and return default empty lists.

### D. [test_traversal.py](../../core/tests/test_traversal.py)
- **Goal**: Validates diagnostic traversal math and information gain.
- **Checks**:
  - Capped score math boundaries (`capped_add` bounds at $\pm 2.0$).
  - Safety bounds of logarithmic scales (`safe_log`).
  - Softmax conversions to output probabilities summing to $1.0$.
  - Entropy variance calculations in `_compute_discriminating_evidence` under a mocked NetworkX Graph.

---

## 3. Test Execution Commands

### A. Standard Runner (Built-in)
Run the test runner from the root folder to execute tests silently:
```bash
python -m unittest discover -s core/tests -p "test_*.py"
```

### B. Verbose Runner (Built-in)
Print the name of each test case and its status:
```bash
python -m unittest discover -s core/tests -p "test_*.py" -v
```

### C. Custom Markdown Report Runner (Recommended)
Executes the test suite, prints console details, and generates a structured report file [test_report.md](../../results/test_report.md):
```bash
python scripts/run_unit_tests.py
```