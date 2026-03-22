"""
traversal_fixed.py - CORRECTED VERSION

FIXED ISSUES:
1. Line 310: Changed context = self._get_evidence_context(parent) 
             to context = self._get_evidence_context(evidence)
2. Added fallback logic when Parser returns empty
3. Added answer parsing helper
"""

import math
import json
from core.parser import Parser

class KG_Traversal(Parser):
    # ============ CONFIG ============
    SMOOTH = 1e-6
    MAX_DELTA = 2.0
    ABSENCE_PROB_THRESHOLD = 0.5
    ABSENCE_WEIGHT = 0.5

    def __init__(self, G, scores, user_input=None, nlu_metadata=None):
        Parser.__init__(self)

        self.G = G
        self.scores = scores
        self.nlu_metadata = nlu_metadata

        self.asked = set()
        self.observed_yes = set()
        self.observed_no = set()
        
        # Load evidence JSON for display
        with open('Data/ddxplus/release_evidences.compact.json', 'r') as f:
            self.release_evidences = json.load(f)
    
        if user_input:
            self._parse_initial_query(user_input)
            
    def _get_evidence_context(self, evidence_id):
        """Get JSON context for an evidence"""
        if evidence_id in self.release_evidences:
            return json.dumps([self.release_evidences[evidence_id]])
        return json.dumps([])

    def _get_evidence_info(self, evidence_id):
        """
        Get human-readable information about an evidence from release_evidences.json
        """
        if evidence_id in self.release_evidences:
            evidence = self.release_evidences[evidence_id]
            return {
                'id': evidence_id,
                'code_question': evidence.get('code_question', evidence_id),
                'question_en': evidence.get('question_en', 'Unknown question'),
                'question_fr': evidence.get('question_fr', ''),
                'data_type': evidence.get('data_type', 'B'),
                'possible_values': evidence.get('possible-values', []),
                'is_antecedent': evidence.get('is_antecedent', False),
                'value_meaning': evidence.get('value_meaning', {})
            }
        return None

    def print_evidence_details(self, evidence_id, value=None, status=None):
        """Pretty print evidence details"""
        evidence_info = self._get_evidence_info(evidence_id)
        
        if not evidence_info:
            print(f"  ❌ Evidence {evidence_id} not found in knowledge base")
            return
        
        print(f"\n  📋 Evidence: {evidence_id}")
        print(f"     Code Question: {evidence_info['code_question']}")
        print(f"     Question: {evidence_info['question_en']}")
        print(f"     Type: {evidence_info['data_type']}")
        
        if status:
            print(f"     Answer: {status}")
        
        if value:
            print(f"     Value: {value}")
        
        if evidence_info['is_antecedent']:
            print(f"     📌 Note: This is an antecedent (historical) condition")
        
        if evidence_info['possible_values']:
            print(f"     Possible values: {evidence_info['possible_values']}")

    def _parse_initial_query(self, user_input):
        """Parse free-text user input"""
        context = json.dumps([])
        
        evidences, values = self.parse_query(user_input, context)

        if not evidences:
            print("No evidence successfully parsed from initial input.")
            return

        print("\n📝 Parsed Initial Evidence:")
        for e, v in zip(evidences, values):
            status = "NO" if v == "NO" else "YES"
            self.print_evidence_details(e, value=v, status=status)

        self.apply_initial_evidence(evidences, values)

    def _get_empty_context(self):
        """Get a minimal context JSON for parser"""
        return json.dumps([])

    def _parse_answer_to_yes_no(self, answer_text):
        """Convert user answer to clear YES/NO"""
        answer = answer_text.lower().strip()
        
        if any(w in answer for w in ["yes", "yep", "sure", "definitely", "always", "often"]):
            return "YES"
        elif any(w in answer for w in ["no", "nope", "never", "not at all", "not really"]):
            return "NO"
        else:
            return "NO"  # Default to NO if unclear

    # ============ UTIL ============
    def safe_log(self, p):
        return math.log(max(p, self.SMOOTH))

    def capped_add(self, score, delta):
        return score + max(-self.MAX_DELTA, min(self.MAX_DELTA, delta))

    # ============ EVIDENCE SELECTION ============
    def get_discriminating_evidence(self, candidate_conditions):
        """Select the most informative evidence question to ask next"""
        best_e, best_gain = None, -1.0

        max_s = max(self.scores[c] for c in candidate_conditions)
        post = {
            c: math.exp(self.scores[c] - max_s)
            for c in candidate_conditions
        }
        Z = sum(post.values())
        post = {c: p / Z for c, p in post.items()}

        for c in candidate_conditions:
            for _, e, _ in self.G.out_edges(c, data=True):
                if self.G.nodes[e]["type"] != "evidence":
                    continue

                if e in self.asked:
                    continue

                parent = self.G.nodes[e].get("parent", None)

                # Skip children if parent was answered NO
                if parent and parent in self.observed_no:
                    continue

                ps = []
                for c2 in candidate_conditions:
                    p = (
                        self.G.edges[c2, e]["p_e_given_c"]
                        if self.G.has_edge(c2, e)
                        else self.SMOOTH
                    )
                    ps.append((post[c2], p))

                mean = sum(w * p for w, p in ps)
                gain = sum(w * (p - mean) ** 2 for w, p in ps)

                if gain > best_gain:
                    best_gain, best_e = gain, e

        return best_e

    # ============ SCORE UPDATES ============
    def apply_binary_answer(self, evidence, is_yes):
        """Update condition scores based on binary yes/no answer"""
        for c in self.scores:
            p = (
                self.G.edges[c, evidence]["p_e_given_c"]
                if self.G.has_edge(c, evidence)
                else self.SMOOTH
            )

            if is_yes:
                self.scores[c] = self.capped_add(self.scores[c], self.safe_log(p))
            else:
                if p >= self.ABSENCE_PROB_THRESHOLD:
                    penalty = self.safe_log(1.0 - p)
                    self.scores[c] = self.capped_add(
                        self.scores[c],
                        self.ABSENCE_WEIGHT * penalty
                    )

    def apply_value_answer(self, evidence, chosen_values):
        """Update condition scores based on categorical/numerical answer"""
        for c in self.scores:
            best_pv = self.SMOOTH
            for v in chosen_values:
                stats = self.G.edges[evidence, v].get("cond_stats", {})
                best_pv = max(
                    best_pv,
                    stats.get(c, {}).get("p_v_given_e_c", self.SMOOTH)
                )
            self.scores[c] = self.capped_add(self.scores[c], self.safe_log(best_pv))

    def apply_initial_evidence(self, evidences, values):
        """Apply evidence to scores"""
        for evid, val in zip(evidences, values):
            self.asked.add(evid)

            # Ensure parents are marked as asked if a child is triggered
            parent = self.G.nodes[evid].get("parent")
            if val not in ("YES", "NO") and parent:
                self.asked.add(parent)

            # Track observations
            if val == "NO":
                self.observed_no.add(evid)
            else:
                self.observed_yes.add(evid)

            # Apply mathematical scoring
            if val not in ("YES", "NO"):
                self.apply_value_answer(evid, val)
            else:
                self.apply_binary_answer(evid, val == "YES")

    # ============ MAIN LOOP ============
    def run(self, max_steps=5, top_k_conditions=10):
        """Interactive diagnostic loop with Bayesian updates"""
        print("\n=== INTERACTIVE DIAGNOSTIC TRAVERSAL ===")

        step = 0
        while step < max_steps:       
            # --- A. Rank and Display the current Top Candidates ---
            ranked = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)[:top_k_conditions]
            candidate_conditions = [c for c, _ in ranked]

            print(f"\nStep {step + 1} — Current Differential:")
            for c, s in ranked:
                print(f"  {c:40s} score={s:.4f}")

            if len(candidate_conditions) <= 1:
                break

            # --- B. Decide the next Question ---
            evidence = self.get_discriminating_evidence(candidate_conditions)
            if evidence is None:
                break

            parent = self.G.nodes[evidence].get("parent", None)
            
            # --- Handle Parent Evidence if exists ---
            if parent and parent != evidence and parent not in self.observed_yes:
                if parent not in self.asked:
                    question = self.G.nodes[parent]['question_en']
                    
                    print(f"\n{'='*70}")
                    print(f"SCREENING QUESTION (Parent Evidence)")
                    print(f"{'='*70}")
                    self.print_evidence_details(parent)
                    
                    print(f"\n🩺 Question: {question}")
                    ans = input("Your answer: ").strip()

                    # CORRECT: Send context for parent (we're asking about parent)
                    context = self._get_evidence_context(parent)
                    context_text = f"The doctor asked: '{question}'. The patient answered: '{ans}'"
                    ext_evidences, ext_values = self.parse_query(context_text, context)

                    # FALLBACK: If Parser returns empty, apply answer directly
                    if not ext_evidences:
                        print(f"\n⚠️  Parser returned empty. Applying answer directly to {parent}")
                        ext_evidences = [parent]
                        ext_values = [self._parse_answer_to_yes_no(ans)]
                    
                    if ext_evidences:
                        print(f"\n✅ Processing your answer...")
                        for e, v in zip(ext_evidences, ext_values):
                            self.print_evidence_details(e, value=v, status=v)
                        
                        self.apply_initial_evidence(ext_evidences, ext_values)
                    
                    self.asked.add(parent)
                    step += 1

                if parent in self.observed_no:
                    continue

            # --- Handle Chosen Evidence ---
            self.asked.add(evidence)
            question = self.G.nodes[evidence]['question_en']
            
            print(f"\n{'='*70}")
            print(f"MAIN DIAGNOSTIC QUESTION")
            print(f"{'='*70}")
            self.print_evidence_details(evidence)
            
            print(f"\n🩺 Question: {question}")
            ans = input("Your answer: ").strip()

            # ✅ FIXED: Send context for 'evidence' (the one we're asking about), not 'parent'!
            context = self._get_evidence_context(evidence)
            context_text = f"The doctor asked: '{question}'. The patient answered: '{ans}'"
            ext_evidences, ext_values = self.parse_query(context_text, context)
            
            # FALLBACK: If Parser returns empty, apply answer directly
            if not ext_evidences:
                print(f"\n⚠️  Parser returned empty. Applying answer directly to {evidence}")
                ext_evidences = [evidence]
                ext_values = [self._parse_answer_to_yes_no(ans)]
            
            if ext_evidences:
                print(f"\n✅ Processing your answer...")
                for e, v in zip(ext_evidences, ext_values):
                    self.print_evidence_details(e, value=v, status=v)
                
                self.apply_initial_evidence(ext_evidences, ext_values)
            
            step += 1

        print("\n" + "="*70)
        print("=== FINAL RANKED CONDITIONS ===")
        print("="*70)
        for c, s in sorted(self.scores.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"{c:40s} score={s:.4f}")