import math
from core.parser import Parser

# Note: We removed DDxGraphNLU entirely! 
class KG_Traversal(Parser):
    # ---------------- CONFIG ----------------
    SMOOTH = 1e-6
    MAX_DELTA = 2.0
    ABSENCE_PROB_THRESHOLD = 0.5
    ABSENCE_WEIGHT = 0.5

    def __init__(self, G, scores, user_input=None):
        Parser.__init__(self) # Initialize the LangChain LLM Pipeline

        self.G = G
        self.scores = scores

        self.asked = set()
        self.observed_yes = set()
        self.observed_no = set()
    
        if user_input:
            self._parse_initial_query(user_input)

    def _parse_initial_query(self, user_input):
        """
        Parse free-text user input and initialize evidence + scores using Langchain parser.
        """
        evidences, values = self.parse_query(user_input)

        if not evidences:
            print("No evidence successfully parsed from initial input.")
            return

        print("\nParsed initial evidence:")
        for e, v in zip(evidences, values):
            print(f"  {e} → {v}")

        self.apply_initial_evidence(evidences, values)

    # ---------------- UTIL ----------------
    def safe_log(self, p):
        return math.log(max(p, self.SMOOTH))

    def capped_add(self, score, delta):
        return score + max(-self.MAX_DELTA, min(self.MAX_DELTA, delta))

    # ---------------- EVIDENCE SELECTION ----------------
    def get_discriminating_evidence(self, candidate_conditions):
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
                if self.G.nodes[e]["type"] != "evidence" or e in self.asked:
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

    # ---------------- SCORE UPDATES ----------------
    def apply_binary_answer(self, evidence, is_yes):
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
        """
        Unified logic to apply evidence to the graph. 
        Works for both the initial input AND the interactive LLM loop.
        """
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
                self.observed_yes.add(evid) # Both 'YES' and value arrays count as presence

            # Apply mathematical scoring
            if val not in ("YES", "NO"):
                self.apply_value_answer(evid, val)
            else:
                self.apply_binary_answer(evid, val == "YES")

    # ---------------- MAIN LOOP ----------------
    def run(self, max_steps=5, top_k_conditions=10):
        print("\n=== INTERACTIVE DIAGNOSTIC TRAVERSAL ===")

        for step in range(max_steps):
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
            if evidence is None: break

            parent = self.G.nodes[evidence].get("parent", None)
            
            # --- Handle Parent Evidence if exists ---
            if parent and parent != evidence and parent not in self.observed_yes:
                if parent not in self.asked:
                    question = self.G.nodes[parent]['question_en']
                    print(f"\n🩺 Question: {question}")
                    ans = input("Your answer: ").strip()

                    # Feed both the doctor's question and patient's answer to the LLM
                    context_text = f"The doctor asked: '{question}'. The patient answered: '{ans}'"
                    ext_evidences, ext_values = self.parse_query(context_text)

                    if ext_evidences:
                        self.apply_initial_evidence(ext_evidences, ext_values)
                    
                    self.asked.add(parent)

                if parent in self.observed_no:
                    continue

            # --- Handle Chosen Evidence ---
            self.asked.add(evidence)
            question = self.G.nodes[evidence]['question_en']
            
            print(f"\n🩺 Question: {question}")
            ans = input("Your answer: ").strip()

            # Feed both the doctor's question and patient's answer to the LLM
            context_text = f"The doctor asked: '{question}'. The patient answered: '{ans}'"
            ext_evidences, ext_values = self.parse_query(context_text)
            
            if ext_evidences:
                self.apply_initial_evidence(ext_evidences, ext_values)

        print("\n=== FINAL RANKED CONDITIONS ===")
        for c, s in sorted(self.scores.items(), key=lambda x: x[1], reverse=True):
            print(f"{c:40s} score={s:.4f}")