import math
from core.nlu import DDxGraphNLU
from core.parser import Parser


class KG_Traversal(Parser, DDxGraphNLU):
    # ---------------- CONFIG ----------------
    SMOOTH = 1e-6
    MAX_DELTA = 2.0
    ABSENCE_PROB_THRESHOLD = 0.5
    ABSENCE_WEIGHT = 0.5

    def __init__(self, G, scores, user_input=None):
        Parser.__init__(self)  # Initialize the LangChain LLM Pipeline
        DDxGraphNLU.__init__(self, G)  # Initialize the DDxGraphNLU

        self.G = G
        self.scores = scores

        self.asked = set()
        self.observed_yes = set()
        self.observed_no = set()

        if user_input:
            self._parse_initial_query(user_input)

    def _parse_initial_query(self, user_input):
        """
        Parse free-text user input and initialize evidence + scores.
        """
        context = self.retrieve(user_input)
        evidences, values = self.parse_query(user_input, context)

        if not evidences:
            return

        print("\nParsed initial evidence:")
        for e, v in zip(evidences, values):
            print(f"  {e} → {v}")

        self.apply_initial_evidence(evidences=evidences, values=values)

    # ---------------- UTIL ----------------
    def safe_log(self, p):
        return math.log(max(p, self.SMOOTH))

    def capped_add(self, score, delta):
        return score + max(-self.MAX_DELTA, min(self.MAX_DELTA, delta))

    # ---------------- EVIDENCE SELECTION ----------------
    def get_discriminating_evidence(self, candidate_conditions):
        best_e, best_gain = None, -1.0

        max_s = max(self.scores[c] for c in candidate_conditions)
        post = {c: math.exp(self.scores[c] - max_s) for c in candidate_conditions}
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
                self.observed_yes.add(evidence)
            else:
                if p >= self.ABSENCE_PROB_THRESHOLD:
                    penalty = self.safe_log(1.0 - p)
                    self.scores[c] = self.capped_add(
                        self.scores[c], self.ABSENCE_WEIGHT * penalty
                    )
                self.observed_no.add(evidence)

        self.asked.add(evidence)

    def apply_value_answer(self, evidence, chosen_values):
        self.observed_yes.add(evidence)

        for c in self.scores:
            best_pv = self.SMOOTH
            for v in chosen_values:
                stats = self.G.edges[evidence, v].get("cond_stats", {})
                best_pv = max(
                    best_pv, stats.get(c, {}).get("p_v_given_e_c", self.SMOOTH)
                )

            self.scores[c] = self.capped_add(self.scores[c], self.safe_log(best_pv))

        self.asked.add(evidence)

    # ---------------- INITIAL EVIDENCE ----------------
    def apply_initial_evidence(self, evidences, values):
        for evid, val in zip(evidences, values):
            self.asked.add(evid)

            parent = self.G.nodes[evid].get("parent")
            if val not in ("YES", "NO") and parent:
                self.asked.add(parent)

            if val == "YES":
                self.observed_yes.add(evid)
            elif val == "NO":
                self.observed_no.add(evid)

            # Apply scoring
            if val not in ("YES", "NO"):
                for v in val:
                    for c in self.scores:
                        stats = self.G.edges[evid, v].get("cond_stats", {})
                        p = stats.get(c, {}).get("p_v_given_e_c", self.SMOOTH)
                        self.scores[c] += self.safe_log(p)
            else:
                self.apply_binary_answer(evid, val == "YES")

    # ---------------- MAIN LOOP ----------------
    def run(self, max_steps=5, top_k_conditions=10):
        print("\n=== INTERACTIVE DIAGNOSTIC TRAVERSAL ===")

        # for step in range(max_steps):
        #     ranked = sorted(
        #         self.scores.items(),
        #         key=lambda x: x[1],
        #         reverse=True
        #     )[:top_k_conditions]

        #     candidates = [c for c, _ in ranked]

        #     print(f"\nStep {step + 1}")
        #     for c, s in ranked:
        #         print(f"{c:40s} score={s:.4f}")

        #     if len(candidates) <= 1:
        #         break

        #     evidence = self.get_discriminating_evidence(candidates)
        #     if not evidence:
        #         break

        #     print(f"\n🩺 {self.G.nodes[evidence]['question_en']}")

        #     values = [
        #         v for _, v in self.G.out_edges(evidence)
        #         if self.G.nodes[v]["type"].startswith("possible")
        #     ]

        #     if not values:
        #         ans = input("yes / no: ").strip().lower()
        #         self.apply_binary_answer(evidence, ans in ("y", "yes"))
        #         continue

        #     text = input()
        #     query = self.G.nodes[evidence]['question_en'] + " " + text
        #     matches = self._find_best_match(query)

        #     if matches:
        #         for m in matches:
        #             if m and m["value"]:
        #                 chosen = m["value"] if isinstance(m["value"], list) else [m["value"]]
        #                 self.apply_value_answer(evidence, chosen)

        # print("\n=== FINAL RANKED CONDITIONS ===")
        # for c, s in sorted(self.scores.items(), key=lambda x: x[1], reverse=True):
        #     print(f"{c:40s} score={s:.4f}")

        for step in range(max_steps):
            # --- A. Rank and Display the current Top Candidates ---
            ranked = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)[
                :top_k_conditions
            ]
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

            # print(f"Selected evidence: {self.G.nodes[evidence]}")
            parent = self.G.nodes[evidence].get("parent", None)
            print(f"Parent evidence: {parent}")

            if parent and parent != evidence and parent not in self.observed_yes:
                if parent not in self.asked:
                    question = self.G.nodes[parent]["question_en"]
                    print(f"\n🩺 Question: {question}")
                    ans = input("Your answer: ").strip()

                    # Feed both the doctor's question and patient's answer to the LLM
                    context_text = (
                        f"The doctor asked: '{question}'. The patient answered: '{ans}'"
                    )
                    context = self.retrieve(context_text)
                    ext_evidences, ext_values = self.parse_query(context_text, context)

                    if ext_evidences:
                        self.apply_initial_evidence(ext_evidences, ext_values)

                    self.asked.add(parent)

                if parent in self.observed_no:  # is asked but is not in observed_yes
                    continue
                # if parent is asked and is in observed_yes, we proceed to ask evidence

            self.asked.add(evidence)
            question = self.G.nodes[evidence]["question_en"]

            print(f"\n🩺 Question: {question}")
            ans = input("Your answer: ").strip()

            # Feed both the doctor's question and patient's answer to the LLM
            context_text = (
                f"The doctor asked: '{question}'. The patient answered: '{ans}'"
            )
            context = self.retrieve(context_text)
            ext_evidences, ext_values = self.parse_query(context_text, context)

            if ext_evidences:
                self.apply_initial_evidence(ext_evidences, ext_values)
            step += 1
        print("\n=== FINAL RANKED CONDITIONS ===")
        top_candidates = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k_conditions
        ]
        for c, s in top_candidates:
            print(f"{c:40s} score={s:.4f}")

        top_conditions_with_scores = {c: s for c, s in top_candidates}
        cond_evidence_map = {}
        missing_evidence_map = {}

        for c, _ in top_candidates:
            supporting = []
            for e in self.observed_yes:
                if self.G.has_edge(c, e):
                    supporting.append(e)

            absent = []
            for e in self.observed_no:
                if self.G.has_edge(c, e):
                    p = self.G.edges[c, e].get("p_e_given_c", self.SMOOTH)
                    if p >= self.ABSENCE_PROB_THRESHOLD:
                        absent.append(e)

            cond_evidence_map[c] = supporting
            missing_evidence_map[c] = absent

        return {
            "top_conditions_with_scores": top_conditions_with_scores,
            "cond_evidence_map": cond_evidence_map,
            "missing_evidence_map": missing_evidence_map,
        }
