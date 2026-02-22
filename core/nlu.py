import math
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
# The model name must match the one used to generate the embeddings in your
# Knowledge Graph builder. Using a different model will cause vector mismatch.
EMBEDDING_MODEL = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'

# Thresholds to control the sensitivity of the NLU engine.
# THRESH_EVIDENCE: (0.0 - 1.0)
#   - Minimum similarity score required to assume a user is talking about a
#     specific symptom (Evidence).
#   - Lowering this increases recall (catching subtle symptoms) but risks
#     false positives (matching noise).
THRESH_EVIDENCE = 0.50

# THRESH_VALUE: (0.0 - 1.0)
#   - Minimum similarity score required to confirm a specific detail/value
#     (e.g., matching "Chest" instead of just generic "Pain").
#   - This should be higher than THRESH_EVIDENCE to prevent the system from
#     hallucinating specific details when the user was actually vague.
THRESH_VALUE = 0.65

# ==============================================================================
# CLASS: DDxGraphNLU
# Responsibility: Acts as the semantic search engine for the DDxPlus Knowledge Graph.
# It maps unstructured natural language to structured Graph Nodes (Evidence & Values).
# ==============================================================================
class DDxGraphNLU:
    def __init__(self, G, embedding_model_name=EMBEDDING_MODEL):
        """
        Initialization Block.
        Responsibility:
        1. Load the heavy NLP model (SentenceTransformer).
        2. "Index" the Knowledge Graph by extracting pre-computed embeddings
           from Evidence nodes and stacking them into a matrix.

        Why? This converts the search problem from a slow graph traversal
        into a fast Matrix Multiplication operation.
        """
        self.G = G
        # print(f"Loading Model: {embedding_model_name}...")
        self.model = SentenceTransformer(embedding_model_name)

        # --- INDEXING STAGE ---
        # We prepare a list to hold the ID and Embedding for every Evidence node.
        self.evidence_ids = []
        embeddings_list = []

        # print("Loading embeddings from Graph Evidence Nodes...")
        count = 0

        # Iterate over all nodes in the NetworkX graph
        for node, data in G.nodes(data=True):
            # Filter for nodes that act as "Questions" (Evidences)
            if data.get("type") == "evidence":

                # OPTIMIZATION: Check if embedding exists in the graph.
                # This avoids re-calculating embeddings (which is slow) every time we restart.
                if "embedding" in data:
                    emb = np.array(data["embedding"])
                    embeddings_list.append(emb)
                    self.evidence_ids.append(node)
                    count += 1
                else:
                    # Fallback: If the graph builder forgot to embed this node,
                    # we do it now on-the-fly (slower startup).
                    txt = f"{data.get('question_en', '')} {data.get('question_fr', '')}"
                    emb = self.model.encode(txt)
                    embeddings_list.append(emb)
                    self.evidence_ids.append(node)

        # Convert list of vectors into a 2D Numpy Matrix for vectorized cosine_similarity
        if count > 0:
            self.evidence_matrix = np.vstack(embeddings_list)
            # print(f" -> Loaded {len(self.evidence_ids)} evidence vectors.")
        else:
            print("Warning: No pre-computed embeddings found. Engine might be slow.")

    def parse_query(self, user_query):
        """
        Main Processing Pipeline.
        Responsibility:
        1. Breaks complex sentences into atomic "Chunks" (Span Chunking).
        2. Detects negation logic (e.g., "no fever").
        3. Orchestrates the search for each chunk and aggregates findings.

        Args:
            user_query (str): The raw input string from the user.

        Returns:
            list: A list of result dictionaries containing 'eid', 'value', 'negated', etc.
        """
        all_findings = []

        # --- STEP 1: SPAN CHUNKING ---
        # We split the user's text into logical units based on punctuation and conjunctions.
        # The Regex `(?<!\d)\.(?!\d)` splits on periods but PROTECTS decimals (e.g., "39.5").
        delimiters = r'[,;]|\bbut\b|\band\b|\balso\b|\bhowever\b|\bplus\b|\bexcept\b|(?<!\d)\.(?!\d)'
        raw_chunks = [c.strip() for c in re.split(delimiters, user_query.lower()) if c.strip()]

        print(f"\n[NLU] Processing: '{user_query}'")

        for i, chunk in enumerate(raw_chunks):
            # # Skip very short chunks that are likely noise
            # if len(chunk) < 3: continue
            print(f" -> Chunk {i+1}: '{chunk}'")

            # --- STEP 2: NEGATION CHECK ---
            # Simple heuristic to see if the chunk contains negation words.
            # If true, we flag the result so the Diagnosis Engine knows to treat it as "Absent".
            is_negated = any(n in chunk for n in ["no ", "not ", "don't ", "never ", "without "])

            # --- STEP 3: SEMANTIC SEARCH ---
            # Run the core matching logic on this specific chunk
            matches = self._find_best_match(chunk)

            if matches:
                for match in matches:
                    if is_negated: match['negated'] = True
                    all_findings.append(match)

        evidences = []
        values = []
        # pattern = re.compile(r'^(?P<eid>E_\d+)_@_(?P<vid>V_\d+|\d+)$')

        for res in all_findings:
            evidences.append(res['eid'])
            status = "NO" if res.get('negated') else "YES"
            if res['value'] and status == "YES":
                # match = pattern.match(res['value'])
                # if match:
                values.append(res['value'])
            else:
                values.append(status)

        return all_findings, evidences, values

    def _find_best_match(self, text, top_k=5):
            query_vec = self.model.encode([text])
            scores = cosine_similarity(query_vec, self.evidence_matrix)[0]

            # Get Top-K Candidates
            top_indices = np.argsort(scores)[::-1][:top_k]

            best_results = []
            best_global_score = -1.0

            print(f"    [Debug] Checking Top {top_k} Candidates for span: '{text}'")

            for idx in top_indices:
                eid = self.evidence_ids[idx]
                e_score = scores[idx]

                evidence_node = self.G.nodes[eid]
                dtype = evidence_node.get("data_type", "B")

                # Initialize candidate with the Question Score
                current_candidate = {
                    'eid': eid,
                    'value': [],
                    'data_type': dtype,
                    'score': e_score,
                    'match_type': 'question_match'
                }

                # --- STAGE 2: VALUE CHECK (The Fix) ---
                # Check values BEFORE applying the threshold filter.
                # This allows low-scoring questions to be saved by high-scoring answers.
                if dtype in ["M", "C"]:
                    val_nodes = [v for _, v in self.G.out_edges(eid)
                                if self.G.nodes[v].get("type", "").startswith("possible")]

                    if val_nodes:
                        val_ids = []
                        val_embs = []
                        for v in val_nodes:
                            v_data = self.G.nodes[v]
                            if "embedding" in v_data:
                                val_ids.append(v)
                                val_embs.append(np.array(v_data["embedding"]))

                        if val_embs:
                            val_matrix = np.vstack(val_embs)
                            v_scores = cosine_similarity(query_vec, val_matrix)[0]
                            for i, vs in enumerate(v_scores): 
                                # BOOST SCORE if value is better
                                if vs > current_candidate['score']:
                                    current_candidate['score'] = vs
                                    current_candidate['match_type'] = 'value_match'

                                    # Only lock in the value if it's confident enough
                                    if vs >= THRESH_VALUE:
                                        current_candidate['value'].append(val_ids[i])

                # --- LATE FILTERING ---
                # NOW we check if the (potentially boosted) score meets the threshold
                if current_candidate['score'] < THRESH_EVIDENCE:
                    # Debug print to see what we skipped
                    # print(f"      -> Skipped {eid} (Score {current_candidate['score']:.3f} < {THRESH_EVIDENCE})")
                    continue

                print(f"      -> Candidate {eid}: RawQ={e_score:.3f}, FinalScore={current_candidate['score']:.3f}")

                best_results.append(current_candidate)
                if current_candidate['score'] > best_global_score:
                    best_global_score = current_candidate['score']

            sorted_results = sorted(best_results, key=lambda x: x['score'], reverse=True)
            k = 0
            for res in sorted_results:
                if best_global_score - res['score'] > 0.05:
                    break
                k += 1
            
            return sorted_results[:k] if k > 0 else None


# ==============================================================================
# MAIN EXECUTION BLOCK (Usage Example)
# ==============================================================================
if __name__ == "__main__":
    # Assumes 'G' is a NetworkX graph already loaded with DDxPLUS data
    # and embeddings on the nodes.
    G = pickle.load(open("Pickle/kg.pkl", "rb"))
    # 1. Initialize Engine
    nlu = DDxGraphNLU(G)

    # 2. Define a complex user query with multiple symptoms and negation
    user_input = "I have no pain"

    # 3. Run the Parser
    results, evidence, values = nlu.parse_query(user_input)

    # 4. Display Results
    print(f"\n=== RESULT: Found {len(results)} items ===")
    for res in results:
        status = "NO" if res.get('negated') else "YES"
        if res['value']:
            val_str = "= "
            for v in res['value']:
                val_str += f"{v}, "
            val_str = val_str.rstrip(", ")
        else:
            val_str = ""
        print(f"Evidence: {res['eid']}{val_str} [{status}]")

    for evid, val in zip(evidence, values):
        print(f"{evid}: {val}")