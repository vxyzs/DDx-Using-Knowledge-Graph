# Design Document: Natural Language Understanding (NLU) Retriever

The file [nlu.py](file:///c:/Users/91897/OneDrive/Desktop/FYP/DDx-Using-Knowledge-Graph/core/nlu.py) implements the class `DDxGraphNLU`, which inherits from `BaseSymptomRetriever`. It acts as the semantic similarity search engine that maps natural language queries to structured symptoms in the Knowledge Graph.

---

## 1. Constants & Threshold Settings

- **`EMBEDDING_MODEL`**: `'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'`
  - *Rationale*: A domain-specific BERT model pre-trained on biomedical text. This model must match the model used by the Knowledge Graph builder, otherwise cosine similarity will yield incorrect results due to mismatched representation spaces.
- **`THRESH_EVIDENCE`**: `0.40`
  - *Rationale*: Represents the minimum cosine similarity score needed to assume a user is speaking about a symptom. Setting this lower catches subtle mentions but increases the risk of false positives; setting it higher reduces noise but might miss vague/informal descriptions.
- **`THRESH_VALUE`**: `0.55`
  - *Rationale*: The minimum similarity score to lock in a specific sub-value (e.g., matching "around my eye and temple" to a location value node, rather than just matching the parent symptom node). This threshold is higher than `THRESH_EVIDENCE` to prevent the system from registering specific details unless they are explicitly and clearly mentioned.

---

## 2. Initialization and Indexing

On initialization (`__init__`), the NLU retriever performs the following steps:
1. **Model Loading**: Loads the `SentenceTransformer` model (using the specified `EMBEDDING_MODEL` name).
2. **Knowledge Graph Indexing**:
   - Iterates through the NetworkX Graph `G` to locate all nodes where `type == "evidence"`.
   - **Optimization Check**: If `embedding` is already present inside the graph node, it loads it directly (preventing extremely slow re-generation on system startup).
   - **On-the-fly Fallback**: If the graph builder omitted the embedding, it generates it on the fly by combining English and French questions: `f"{question_en} {question_fr}"`.
   - Stack all loaded/generated embeddings into a single 2D NumPy array (`self.evidence_matrix`) to enable fast, vectorized matrix-based cosine similarity computations.

---

## 3. The Query Processing Pipeline (`parse_query1`)

When a user submits a query, it flows through three main phases:

### Phase A: Span Chunking
Splits the user's complex sentence into logical units based on punctuation and conjunctions.
- **Delimiter Regex**: `r'[,;]|\bbut\b|\band\b|\balso\b|\bhowever\b|\bplus\b|\bexcept\b|(?<!\d)\.(?!\d)'`
- *Rationale*: Splitting allows the similarity matcher to evaluate individual symptoms separately (e.g., separating "I have fever" and "but no cough"). The negative lookbehind/lookahead `(?<!\d)\.(?!\d)` prevents decimal values like body temperature "39.5" from triggering a split.

### Phase B: Negation Check
Checks if the chunk contains negative words: `["no ", "not ", "don't ", "never ", "without "]`.
- *Rationale*: Flags the symptom as negated so the diagnostic engine knows to treat it as "Absent" (`"NO"`) instead of "Present" (`"YES"`).

### Phase C: Semantic Search
Invokes the internal matcher (`_find_best_match`) for each chunk, compiling matched evidence IDs and their corresponding values.

---

## 4. Semantic Matching Heuristics (`_find_best_match`)

The core similarity search uses a multi-stage heuristic:
1. **Candidate Retrieval**: Encodes the chunk and calculates cosine similarities against `self.evidence_matrix`. Identifies the `top_k` (default `7`) best-matching symptom nodes.
2. **Categorical / Multi-Choice (M/C) Value Boosting**:
   - For categorical/multi-choice symptoms, the retriever queries the graph's out-edges to retrieve possible value nodes (e.g., location, severity).
   - If value embeddings are pre-computed, it computes the cosine similarity between the query chunk and each value node's embedding.
   - **Boost score**: If a value node similarity score is higher than the parent question node's score, the candidate's score is boosted to this higher value. This allows a vague/low-scoring parent question to be rescued by a highly confident specific answer match.
   - Values exceeding `THRESH_VALUE` are locked in and added to the candidate.
3. **Late Threshold Filtering**: Only retains candidates whose finalized scores meet or exceed `THRESH_EVIDENCE`.
4. **Relative Threshold Filtering**: Re-sorts candidates. It filters out any matches that are more than `0.1` below the score of the highest candidate. This prevents low-scoring, unrelated symptoms from polluting the retrieval list when a single symptom is extremely dominant.
