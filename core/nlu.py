"""
nlu_with_fallback.py - ENHANCED NLU with Smart Fallback

Key improvements:
1. Return ~30 candidate evidences with confidence scores
2. If NO matches found: Send ALL evidences to Parser (let LLM decide)
3. If WEAK matches: Send top candidates + possible values for reasoning
4. Flexible thresholds - don't rigidly filter

Goal: NLU pre-filters 200+ evidences to ~30 for Parser
      If NLU fails (no match), send context so Parser can still reason
"""

import json
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle

with open('Data/ddxplus/release_evidences.compact.json', 'r') as f:
    release_evidences = json.load(f)

# ==============================================================================
# FLEXIBLE CONFIGURATION
# ==============================================================================
EMBEDDING_MODEL = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'

THRESH_EVIDENCE_SOFT = 0.35  # Soft threshold - candidates below this are risky
THRESH_VALUE_SOFT = 0.50     # Soft threshold for values

TOP_K_PER_CHUNK = 10         # Return top 10 per chunk
MAX_TOTAL_FINDINGS = 35      # Cap total findings at ~35
MIN_FINDINGS_BEFORE_FALLBACK = 3  # If < 3 findings, trigger fallback


# ==============================================================================
# CLASS: DDxGraphNLU - ENHANCED WITH FALLBACK
# ==============================================================================
class DDxGraphNLU:
    def __init__(self, G, embedding_model_name=EMBEDDING_MODEL):
        """
        Initialize NLU with flexible candidate selection + smart fallback.
        Goal: Return ~30 best candidates, or ALL candidates if no matches
        """
        with open('Data/ddxplus/release_evidences.compact.json', 'r') as f:
            self.release_evidences = json.load(f)
        self.G = G
        
        print(f"[NLU Init] Loading embedding model: {embedding_model_name}...")
        self.model = SentenceTransformer(embedding_model_name)

        # --- INDEXING STAGE ---
        self.evidence_ids = []
        self.evidence_questions = {}  # Map EID → question text
        embeddings_list = []

        print("[NLU Init] Indexing graph evidence nodes...")
        count = 0

        for node, data in G.nodes(data=True):
            if data.get("type") == "evidence":
                if "embedding" in data:
                    emb = np.array(data["embedding"])
                    embeddings_list.append(emb)
                    self.evidence_ids.append(node)
                    self.evidence_questions[node] = data.get('question_en', '')
                    count += 1
                else:
                    txt = f"{data.get('question_en', '')} {data.get('question_fr', '')}"
                    emb = self.model.encode(txt)
                    embeddings_list.append(emb)
                    self.evidence_ids.append(node)
                    self.evidence_questions[node] = data.get('question_en', '')

        if count > 0:
            self.evidence_matrix = np.vstack(embeddings_list)
            print(f"[NLU Init] ✓ Indexed {len(self.evidence_ids)} evidence nodes")
        else:
            print("[NLU Init] ⚠ Warning: No pre-computed embeddings found.")

    def parse_query(self, user_query, verbose=True):
        """
        Parse query and return ~30 best candidates.
        If no matches found, trigger fallback (send all evidence context).
        """
        all_findings = []

        # --- STEP 1: SPAN CHUNKING ---
        delimiters = r'[,;]|\bbut\b|\band\b|\balso\b|\bhowever\b|\bplus\b|\bexcept\b|(?<!\d)\.(?!\d)'
        raw_chunks = [c.strip() for c in re.split(delimiters, user_query.lower()) if c.strip()]

        if verbose:
            print(f"\n[NLU] Processing: '{user_query}'")
            print(f"[NLU] Found {len(raw_chunks)} chunks")

        for i, chunk in enumerate(raw_chunks):
            if verbose:
                print(f" → Chunk {i+1}: '{chunk}'")

            # --- STEP 2: NEGATION CHECK ---
            is_negated = any(n in chunk for n in ["no ", "not ", "don't ", "never ", "without ", "none"])

            # --- STEP 3: SEMANTIC SEARCH (FLEXIBLE) ---
            matches = self._find_best_match_flexible(chunk, verbose=verbose)

            if matches:
                for match in matches:
                    if is_negated:
                        match['negated'] = True
                    all_findings.append(match)

        # Construct output
        evidences = []
        values = []

        for res in all_findings:
            evidences.append(res['eid'])
            status = "NO" if res.get('negated') else "YES"
            if res['value'] and status == "YES":
                values.append(res['value'])
            else:
                values.append(status)

        return all_findings, evidences, values

    def _find_best_match_flexible(self, text, verbose=True):
        """
        FLEXIBLE: Return top K candidates per chunk.
        Don't filter rigidly - let Parser decide what's relevant.
        """
        query_vec = self.model.encode([text])
        scores = cosine_similarity(query_vec, self.evidence_matrix)[0]

        # Get top K candidates (flexible)
        top_indices = np.argsort(scores)[::-1][:TOP_K_PER_CHUNK]

        best_results = []
        best_global_score = -1.0

        if verbose:
            print(f"    [Debug] Top {TOP_K_PER_CHUNK} candidates for span: '{text}'")

        for idx in top_indices:
            eid = self.evidence_ids[idx]
            e_score = scores[idx]

            evidence_node = self.G.nodes[eid]
            dtype = evidence_node.get("data_type", "B")

            # Initialize candidate
            current_candidate = {
                'eid': eid,
                'value': [],
                'data_type': dtype,
                'score': e_score,
                'match_type': 'question_match'
            }

            # --- VALUE CHECK (FLEXIBLE) ---
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

                                # FLEXIBLE: Use soft threshold
                                if vs >= THRESH_VALUE_SOFT:
                                    current_candidate['value'].append(val_ids[i])

            # --- FLEXIBLE FILTERING ---
            if current_candidate['score'] < 0.30:  # Very low - skip
                continue

            if verbose:
                threshold_status = ""
                if current_candidate['score'] >= THRESH_EVIDENCE_SOFT:
                    threshold_status = "✓"  # Good
                elif current_candidate['score'] >= 0.30:
                    threshold_status = "~"  # Borderline (let Parser decide)
                else:
                    threshold_status = "✗"  # Skip
                
                print(f"      → {eid}: RawQ={e_score:.3f}, FinalScore={current_candidate['score']:.3f}, Type={current_candidate['match_type']} {threshold_status}")

            best_results.append(current_candidate)
            if current_candidate['score'] > best_global_score:
                best_global_score = current_candidate['score']

        # FLEXIBLE: Return ALL candidates that passed soft threshold
        sorted_results = sorted(best_results, key=lambda x: x['score'], reverse=True)
        
        return sorted_results  # Return all, let Parser filter!

    def retrieve(self, text: str, use_fallback=True):
        """
        Original retrieve method with FALLBACK strategy.
        
        Args:
            text: User input
            use_fallback: If True, on low matches, include all possible values
        """
        _, evidence, values = self.parse_query(text, verbose=False)
        collected_evidences = []
        seen_evidences = set()
        for evid, val in zip(evidence, values):
            if evid not in seen_evidences:
                seen_evidences.add(evid)
                collected_evidences.append(release_evidences[evid])
        
        # FALLBACK: If very few matches, add more context
        if use_fallback and len(collected_evidences) < MIN_FINDINGS_BEFORE_FALLBACK:
            print(f"\n[NLU FALLBACK] Only {len(collected_evidences)} matches found. Adding more evidence context...")
            # Add random sample of other evidence for Parser to consider
            all_eids = list(release_evidences.keys())
            remaining = [e for e in all_eids if e not in seen_evidences]
            
            # Add up to 15 more random evidence for Parser to consider
            import random
            additional = random.sample(remaining, min(15, len(remaining)))
            for eid in additional:
                collected_evidences.append(release_evidences[eid])
            print(f"[NLU FALLBACK] Added {len(additional)} additional evidence for context")
        
        return json.dumps(collected_evidences)

    def retrieve_enhanced(self, text: str, verbose=True, use_fallback=True):
        """
        ENHANCED: Returns ~30 candidates with metadata.
        With FALLBACK: If no matches, sends all possible values for that evidence.
        
        Returns:
            dict: {evidences, values, raw_findings, compact_json, metadata, fallback_triggered}
        """
        raw_findings, evidences, values = self.parse_query(text, verbose=verbose)
        
        # Check if fallback should be triggered
        fallback_triggered = False
        fallback_strategy = "none"
        
        if len(raw_findings) < MIN_FINDINGS_BEFORE_FALLBACK:
            fallback_triggered = True
            fallback_strategy = "insufficient_matches"
            
            if verbose:
                print(f"\n[NLU] ⚠️  FALLBACK TRIGGERED: Only {len(raw_findings)} matches")
                print(f"[NLU] Strategy: {fallback_strategy}")
                print(f"[NLU] Sending all evidence context to Parser for reasoning")
        
        # Get compact JSON
        compact_json = json.loads(self.retrieve(text, use_fallback=use_fallback))

        # Cap to MAX_TOTAL_FINDINGS
        if len(raw_findings) > MAX_TOTAL_FINDINGS:
            sorted_findings = sorted(
                raw_findings,
                key=lambda x: x.get('score', 0),
                reverse=True
            )[:MAX_TOTAL_FINDINGS]
            raw_findings = sorted_findings
            
            # Rebuild evidences and values
            evidences = [f['eid'] for f in raw_findings]
            values = []
            for f in raw_findings:
                status = "NO" if f.get('negated') else "YES"
                if f['value'] and status == "YES":
                    values.append(f['value'])
                else:
                    values.append(status)

        return {
            'evidences': evidences,
            'values': values,
            'raw_findings': raw_findings,
            'compact_json': compact_json,
            'metadata': {
                'thresh_evidence_soft': THRESH_EVIDENCE_SOFT,
                'thresh_value_soft': THRESH_VALUE_SOFT,
                'top_k_per_chunk': TOP_K_PER_CHUNK,
                'max_total_findings': MAX_TOTAL_FINDINGS,
                'num_findings': len(raw_findings),
                'unique_evidences': len(set(e for e in evidences)),
            },
            'fallback': {
                'triggered': fallback_triggered,
                'strategy': fallback_strategy,
                'total_context_provided': len(compact_json)
            }
        }


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    G = pickle.load(open("Pickle/kg.pkl", "rb"))
    nlu = DDxGraphNLU(G)

    # Test cases showing fallback
    test_cases = [
        "i have pain in head",
        "pain in upper right chest",
        "i have pain in head and in upper right chest",
        "not really have fever",
        "kinda have pain, not really",
        "xyz123 random text",  # Should trigger fallback (no match)
    ]

    for test_text in test_cases:
        print("\n" + "="*80)
        print(f"TEST: {test_text}")
        print("="*80)
        result = nlu.retrieve_enhanced(test_text, verbose=True)
        
        print(f"\n📊 NLU Results:")
        print(f"   • Evidences: {len(result['evidences'])}")
        print(f"   • Raw findings: {len(result['raw_findings'])}")
        print(f"   • Context provided to Parser: {result['fallback']['total_context_provided']} evidences")
        
        if result['fallback']['triggered']:
            print(f"   • ⚠️  FALLBACK TRIGGERED: {result['fallback']['strategy']}")
        
        if result['raw_findings']:
            print(f"\n   Top findings:")
            for i, finding in enumerate(result['raw_findings'][:5], 1):
                print(f"      {i}. {finding['eid']:12s} │ Score: {finding.get('score', 0):.3f}")