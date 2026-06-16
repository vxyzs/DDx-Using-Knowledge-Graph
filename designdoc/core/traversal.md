# Design Document: Diagnostic Traversal Engine

The file [traversal.py](../../core/traversal.py) implements the foundational traversal classes: the abstract `BaseTraversal` class and the concrete interactive `KG_Traversal` class.

---

## 1. Abstract Base Class (`BaseTraversal`)

`BaseTraversal` centralizes the mathematical scoring formulas, configuration settings, and structural heuristics used by both the interactive console loop and the batch evaluator.

### Configuration Constants
- **`SMOOTH`** (`1e-6`): Added to probabilities when calculating log-likelihoods to prevent division by zero or mathematical domain errors ($\log(0)$).
- **`MAX_DELTA`** (`2.0`): Caps the magnitude of score additions/subtractions in a single step, ensuring that a single symptom match/mismatch does not completely overpower the cumulative scoring history.
- **`ABSENCE_PROB_THRESHOLD`** (`0.5`): Specifies the minimum conditional probability required to penalize a disease for a symptom's absence.
- **`ABSENCE_WEIGHT`** (`0.5`): Scales the penalty for an absent symptom, reflecting that symptom absence is generally less clinically informative than symptom presence.

### Common Heuristics and Math Formulas

#### A. Capped Accumulation
```python
def capped_add(self, score, delta):
    return score + max(-self.MAX_DELTA, min(self.MAX_DELTA, delta))
```

#### B. Discriminating Evidence Selection (`_compute_discriminating_evidence`)
Calculates the expected variance/information gain for each unasked evidence node across candidate conditions.
1. Computes the posterior probability distribution $P(C)$ of candidate conditions by applying softmax over current log scores.
2. For each candidate evidence $E$, retrieves the conditional probability $P(E|C)$ for all candidate conditions.
3. Computes the weighted mean probability of the evidence:
   $$\mu_E = \sum_C P(C) P(E|C)$$
4. Computes the expected variance (information gain):
   $$\text{Gain}_E = \sum_C P(C) (P(E|C) - \mu_E)^2$$
5. Returns the evidence $E$ that maximizes $\text{Gain}_E$.

#### C. Probability Normalization (`_convert_scores_to_probabilities`)
Applies softmax normalization to raw log-likelihood scores:
$$P(C_i) = \frac{e^{S_i - S_{\max}}}{\sum_j e^{S_j - S_{\max}}}$$
Subtracting $S_{\max}$ prevents floating-point overflow during exponentiation.

---

## 2. Interactive Traversal Engine (`KG_Traversal`)

`KG_Traversal` runs the real-time clinical diagnostic dialogue loop.

### Core Processing Functions

#### A. Parsing Initial Complaints (`_parse_initial_query`)
Queries the NLU retriever and Pydantic parser to parse the patient's opening statement, initializing scores using any detected symptoms.

#### B. Parent Gating Heuristic
Before querying a categorical or sub-symptom question, the loop checks the symptom's parent status:
- If a parent symptom (e.g. "Fever") has not been asked yet, the system asks the parent question first.
- If the parent symptom is answered as absent (`"NO"`), the system skips asking any sub-symptoms (e.g. "Fever severity", "Fever duration"), saving diagnostic steps.

#### C. Interactive Refinement Loop (`run`)
1. Ranks and prints current top candidate pathologies.
2. Selects the most discriminating evidence node.
3. Prompts the user for details, feeding the question and user response back through the parser.
4. Adjusts candidate scores:
   - For positive presence: Adds $\log P(E|C)$ to the scores.
   - For negative presence: If $P(E|C) \ge \text{ABSENCE\_PROB\_THRESHOLD}$, adds $\text{ABSENCE\_WEIGHT} \times \log(1 - P(E|C))$ as a penalty.
   - For categorical values: Adds $\log P(V_i|E, C)$ to the scores.
5. normalizes scores and outputs the final differential diagnosis, listing confirmed and missing symptoms for each condition.
