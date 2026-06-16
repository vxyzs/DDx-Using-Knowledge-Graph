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
- **Model**: `openai/gpt-oss-safeguard-20b` (or other config model).
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

## 4. Parser Invocation & Error Handling

1. **Context Serialization**: The retriever output (JSON string) is injected into the LLM prompt.
2. **Double-Pass Printing**: Prints the raw string returned by the LLM, followed by the parsed Pydantic object dict.
3. **Graceful Fallback**: If LLM API connectivity fails or the model returns a schema validation error, it catches the exception and returns an empty `PatientEvidences(evidences=[], values=[])` object rather than crashing the diagnostic loop.
