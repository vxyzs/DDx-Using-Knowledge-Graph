import json
import re
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from core.interfaces import BaseSymptomRetriever

with open("Data/ddxplus/release_evidences.compact.json", "r") as f:
    release_evidences = json.load(f)

EMBEDDING_MODEL = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
THRESH_EVIDENCE = 0.40
THRESH_VALUE = 0.55

class DDxGraphNLU(BaseSymptomRetriever):
    """
    Symptom retrieval engine leveraging SentenceTransformers and cosine similarity.
    Maps unstructured language inputs to structured knowledge graph evidence nodes.
    """

    def __init__(self, G, embedding_model_name=EMBEDDING_MODEL):
        """
        Initialize the retrieval engine with a networkx graph and embedding model.

        Args:
            G (networkx.Graph): Knowledge graph loaded with DDxPlus data and pre-computed embeddings.
            embedding_model_name (str): SentenceTransformer model string for encoding.
        """
        with open('Data/ddxplus/release_evidences.compact.json', 'r') as f:
            self.release_evidences = json.load(f)
        self.G = G
        self.model = SentenceTransformer(embedding_model_name)

        self.evidence_ids = []
        embeddings_list = []
        count = 0

        for node, data in G.nodes(data=True):
            if data.get("type") == "evidence":
                if "embedding" in data:
                    emb = np.array(data["embedding"])
                    embeddings_list.append(emb)
                    self.evidence_ids.append(node)
                    count += 1
                else:
                    txt = f"{data.get('question_en', '')} {data.get('question_fr', '')}"
                    emb = self.model.encode(txt)
                    embeddings_list.append(emb)
                    self.evidence_ids.append(node)

        if count > 0:
            self.evidence_matrix = np.vstack(embeddings_list)
        else:
            print("Warning: No pre-computed embeddings found. Engine might be slow.")

    def parse_query1(self, user_query):
        """
        Parse user query into individual spans, verify negations, and query similarity search.

        Args:
            user_query (str): Raw string describing symptoms.

        Returns:
            Tuple[List[dict], List[str], List[Any]]:
                - List of result match dictionaries.
                - List of parsed evidence IDs.
                - List of parsed values (either 'YES'/'NO' or lists of categorical/numerical keys).
        """
        all_findings = []
        delimiters = r'[,;]|\bbut\b|\band\b|\balso\b|\bhowever\b|\bplus\b|\bexcept\b|(?<!\d)\.(?!\d)'
        raw_chunks = [c.strip() for c in re.split(delimiters, user_query.lower()) if c.strip()]

        print(f"\n[NLU] Processing: '{user_query}'")

        for i, chunk in enumerate(raw_chunks):
            print(f" -> Chunk {i+1}: '{chunk}'")
            is_negated = any(n in chunk for n in ["no ", "not ", "don't ", "never ", "without "])
            matches = self._find_best_match(chunk)

            if matches:
                for match in matches:
                    if is_negated:
                        match['negated'] = True
                    all_findings.append(match)

        evidences = []
        values = []

        for res in all_findings:
            evidences.append(res["eid"])
            status = "NO" if res.get("negated") else "YES"
            if res["value"] and status == "YES":
                values.append(res["value"])
            else:
                values.append(status)

        return all_findings, evidences, values

    def _find_best_match(self, text, top_k=7):
        """
        Locate best-matching candidate evidence nodes in the Graph.

        Args:
            text (str): Sub-phrase text chunk.
            top_k (int): Number of top candidate keys to evaluate.

        Returns:
            List[dict] or None: List of matching candidates sorted by score within a relative threshold,
                                or None if no match is found.
        """
        query_vec = self.model.encode([text])
        scores = cosine_similarity(query_vec, self.evidence_matrix)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        best_results = []
        best_global_score = -1.0

        print(f"    [Debug] Checking Top {top_k} Candidates for span: '{text}'")

        for idx in top_indices:
            eid = self.evidence_ids[idx]
            e_score = scores[idx]

            evidence_node = self.G.nodes[eid]
            dtype = evidence_node.get("data_type", "B")

            current_candidate = {
                'eid': eid,
                'value': [],
                'data_type': dtype,
                'score': e_score,
                'match_type': 'question_match'
            }

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
                            if vs > current_candidate['score']:
                                current_candidate['score'] = vs
                                current_candidate['match_type'] = 'value_match'

                                if vs >= THRESH_VALUE:
                                    current_candidate['value'].append(val_ids[i])

            if current_candidate['score'] < THRESH_EVIDENCE:
                continue

            print(f"      -> Candidate {eid}: RawQ={e_score:.3f}, FinalScore={current_candidate['score']:.3f}")

            best_results.append(current_candidate)
            if current_candidate['score'] > best_global_score:
                best_global_score = current_candidate['score']

        sorted_results = sorted(best_results, key=lambda x: x['score'], reverse=True)
        k = 0
        for res in sorted_results:
            if best_global_score - res['score'] > 0.1:
                break
            k += 1

        return sorted_results[:k] if k > 0 else None

    def retrieve(self, text: str) -> str:
        """
        Retrieve context records for matched symptoms.

        Args:
            text (str): Text describing symptoms.

        Returns:
            str: JSON serialized representation of matching symptoms.
        """
        _, evidence, values = self.parse_query1(text)
        collected_evidences = []
        seen_evidences = set()
        for evid, val in zip(evidence, values):
            print(f"{evid}: {val}")
            if evid in seen_evidences:
                continue
            seen_evidences.add(evid)
            collected_evidences.append(release_evidences[evid])
        return json.dumps(collected_evidences)

if __name__ == "__main__":
    G = pickle.load(open("Pickle/kg.pkl", "rb"))
    nlu = DDxGraphNLU(G)
    user_input = "For the past couple of weeks, I’ve been having sudden episodes of very intense pain on one side of my head, mainly around my eye and temple. The pain feels sharp and unbearable, and when it happens my eye starts watering and my nose feels blocked on the same side. I can’t stay still during these attacks and feel extremely restless. These episodes happen multiple times and often around the same time of day, then completely go away in between.No fever and cough."
    results, evidence, values = nlu.parse_query1(user_input)
    collected_evidences = nlu.retrieve(user_input)
    print(f"\n=== RESULT: Found {len(results)} items ===")
    for res in results:
        status = "NO" if res.get("negated") else "YES"
        val_str = f"= {', '.join(res['value'])}" if res["value"] else ""
        print(f"Evidence: {res['eid']}{val_str} [{status}]")
    print(collected_evidences)