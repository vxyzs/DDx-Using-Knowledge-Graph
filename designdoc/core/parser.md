# Design Document: LLM-Based Dialogue Parser

The file [parser.py](../../core/parser.py) implements the class `Parser`, which inherits from `BaseSymptomParser`. It leverages a Large Language Model (LLM) to extract structured, type-safe symptom entities and values from unstructured text.

---

## 1. Pydantic Target Schema

To guarantee type safety and structural validation, the parser maps LLM outputs directly into a Pydantic schema:

```python
class PatientEvidences(BaseModel):
    evidences: List[str]
    values: List[Union[str, List[str]]]
```

### Constraints
- **`evidences`**: A list of unique evidence codes (e.g., `["E_55", "E_201"]`). It must only include symptoms explicitly mentioned in the input text.
- **`values`**: A parallel list mapped to the evidence codes.
  - Boolean or absent symptoms must resolve to `"YES"` or `"NO"`.
  - Categorical or numerical values must resolve to lists of exact mapped IDs (e.g., `[["E_55_@_V_12"]]`).

---

## 2. LLM Initialization and Parameter Selection

- **Hugging Face Serverless Router**: Connects via LangChain's `ChatOpenAI` wrapper pointing to `https://router.huggingface.co/v1`.
- **Model Configuration**:
  - *Primary Model*: Loaded dynamically from `config.json` via `model_name`.
  - *Fallback Models*: Configurable backup models (`fallback_models`), defaulting to `[]` for custom model configurations.
  - *Timeout*: Set via `request_timeout` (defaults to `30.0` seconds) to avoid thread blocking on network latency.
- **`temperature=0.0`**:
  - *Rationale*: Crucial for clinical extraction. A temperature of `0` minimizes the model's creativity and hallucination rates, forcing it to remain highly deterministic and factual based strictly on the provided context.

---

## 3. Prompt Engineering Rules

The LLM is governed by a detailed prompt template instructing it on the following structural rules:
1. **Omission Rule**: Do not include any symptoms that were not explicitly mentioned.
2. **Boolean Standard**: Map boolean (`"B"`) symptoms to `"YES"` or `"NO"`.
3. **Categorical IDs**: Map sub-categorical selections to standard graph notation: `[EvidenceID]_@_[ValueID]` (e.g., `["E_55_@_V_125"]`).
4. **Numerical Formats**: Map numerical attributes to a list with the numeric string: `[EvidenceID]_@_[Number]` (e.g., `["E_59_@_5"]`).
5. **Parent-Child Logic (Multi-choice `M`)**: If an multi-choice option is selected, the parent evidence question node must also be marked `"YES"` (e.g., if "E_55_@_V_125" is present, parent "E_55" must be mapped to `"YES"`).
6. **Negation Flagging**: If a symptom is mentioned as absent ("no fever"), map its code to `"NO"` to inform the system.

---

## 4. Parser Invocation & Resilient Error Handling

1. **Context Serialization**: The retriever output (JSON string) is injected into the LLM prompt.
2. **Double-Pass Printing**: Prints the raw string returned by the LLM, followed by the parsed Pydantic object dict.
3. **Resilient Retry and Fallback System**:
   - **Tenacity Integration**: Uses `tenacity` to automatically retry failed API requests or outputs that fail Pydantic parsing.
   - **Configurable Attempts**: Retries each model up to `max_retries` attempts, waiting for a fixed `retry_delay` between attempts (both parameters loaded dynamically from configuration).
   - **Alternative Fallback Models**: If the primary model fails completely after all retries, the parser transitions to subsequent models in the `fallback_models` list.
   - **Safe Base Fallback**: Returns a blank `PatientEvidences(evidences=[], values=[])` object only if all candidate models and their respective retries are completely exhausted, protecting the interactive diagnostic engine from crashing.
