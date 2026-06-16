# Design Document: Traversal Evaluation System

The file [test_only_traversal.py](../../scripts/test_only_traversal.py) implements the batch testing and evaluation framework for the differential diagnosis traversal engine.

---

## 1. Evaluation Scenarios

To measure performance across different clinical situations, the evaluator runs three distinct configurations:

### A. Full Information
Uses the complete list of symptoms associated with the patient record in the dataset. This measures the system's ceiling accuracy when all patient history is available.

### B. Partial Information
Simulates a more realistic scenario where only a fraction of the patient's symptoms are initially known.
- **Sampling Ratio**: `PARTIAL_RATIO = 0.50` (50% of symptoms are retained).
- **Selection Heuristic**: Retains the 50% of symptoms that have the highest conditional probability given the true pathology, simulating a patient reporting their most prominent symptoms first.

### C. Hard Cases
Evaluates performance on clinically ambiguous records.
- **Filtering Rule**: A record is classified as a "hard case" if the true pathology's rank in the database's differential diagnosis list is strictly greater than `5` (`rank > 5`).

---

## 2. Core Metrics Calculated

- **Top-k Accuracy**: The percentage of test cases where the true pathology is successfully included in the top $k$ candidates (evaluated at $k \in \{1, 3, 5\}$).
- **Mean Reciprocal Rank (MRR)**: Measures the average reciprocal rank of the true pathology:
  $$\text{MRR} = \frac{1}{N} \sum_{i=1}^N \frac{1}{\text{rank}_i}$$
- **Mean Rank**: The average rank position of the correct pathology.
- **Average Steps**: The average number of follow-up questions asked before the traversal terminates.
- **Probability Analysis**: Track the average probability assigned to the true pathology vs. the top-1 predicted pathology.

---

## 3. Pathology Analysis and JSON Output

The function `evaluate_per_pathology_and_save` performs a granular evaluation:
1. Identifies all unique pathologies present in the test patient dataset.
2. For each pathology, samples up to `1000` patient cases.
3. Runs the evaluation under the **Full Info**, **Partial Info**, and **Hard Cases** configurations.
4. Generates a structured JSON performance profile saved to `results/per_pathology_results.json`.
