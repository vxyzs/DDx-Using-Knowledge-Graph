# Design Document: Automated Traversal Simulator

The file [test_only_traversal.py](file:///c:/Users/91897/OneDrive/Desktop/FYP/DDx-Using-Knowledge-Graph/core/test_only_traversal.py) implements the class `TestOnlyTraversal`, which inherits from `BaseTraversal`. It acts as an automated clinical diagnostic traversal simulator to evaluate system performance against static patient records.

---

## 1. Architectural Purpose

While `KG_Traversal` is built for real-time doctor-patient interactions, `TestOnlyTraversal` is a scriptable testing engine designed to process patient records from the DDxPlus dataset. It simulates answers to diagnostic questions based on pre-recorded dataset values, measuring the system's accuracy and diagnostic efficiency.

---

## 2. Key Methods

### A. Value Extraction Heuristics

#### `get_top_values`
Selects the most statistically relevant sub-values for a given evidence node across the current top candidate conditions.
- *Rationale*: When querying categorical symptoms, the system needs to know which choices are most probable for the candidate diseases, sorting the possible values by conditional probability ($P(V|E, C)$).

#### `find_existing_values`
Filters the list of values recorded in the dataset to find choices matching the active query.
- **Regex**: `VALUE_PATTERN = re.compile(r'^(E_\d+)_@_(V_\d+|\d+)$')`
- *Rationale*: Maps raw dataset values (e.g. `E_55_@_V_125`) to match the format of the evidence node being queried.

### B. Simulation Loop (`run`)

The loop processes a patients symptoms step-by-step without requiring manual console input:
1. **Initialize State**: Populates initial positive/negative symptoms based on the dataset entry.
2. **Select Evidence**: Invokes `_compute_discriminating_evidence` (from the parent class) to choose the next question.
3. **Apply Parent Gating**: Evaluates if the parent symptom is present. If the parent is absent, it penalizes candidate conditions and skips asking sub-questions.
4. **Apply Answer**:
   - If the queried symptom is binary, the simulation checks if it is in the patient's record and applies a positive score boost or absence penalty.
   - If the symptom is categorical, it retrieves the patient's recorded values (via `find_existing_values`) and applies categorical log likelihood score adjustments.
5. **Normalize Scores**: Calls `_convert_scores_to_probabilities` to generate final probabilities and returns the total step count.
