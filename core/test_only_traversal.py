import math
import re

class TestOnlyTraversal:
    # ---------------- CONFIG ----------------
    SMOOTH = 1e-6
    MAX_DELTA = 2.0
    ABSENCE_PROB_THRESHOLD = 0.5
    ABSENCE_WEIGHT = 0.5

    VALUE_PATTERN = re.compile(r'^(E_\d+)_@_(V_\d+|\d+)$')

    def __init__(self, G):
        self.G = G

    # ---------------- UTILS ----------------
    def safe_log(self, p):
        return math.log(max(p, self.SMOOTH))

    def capped_add(self, score, delta):
        return score + max(-self.MAX_DELTA, min(self.MAX_DELTA, delta))

    # ---------------- VALUE SELECTION ----------------
    def get_top_values(self, evidence, candidate_conditions):
        scores = []
        for _, v in self.G.out_edges(evidence):
            if not self.G.nodes[v]["type"].startswith("possible"):
                continue
            stats = self.G.edges[evidence, v].get("cond_stats", {})
            max_p = max(
                stats.get(c, {}).get("p_v_given_e_c", 0.0)
                for c in candidate_conditions
            )
            scores.append((v, max_p))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [v for v, _ in scores]

    def find_existing_values(self, evidences, shown_values, evidence):
        chosen = []
        for ev in evidences:
            m = self.VALUE_PATTERN.match(ev)
            if not m:
                continue
            eid, vid = m.group(1), m.group(2)
            if eid == evidence and vid in shown_values:
                chosen.append(vid)
        return chosen

    # ---------------- DISCRIMINATING EVIDENCE ----------------
    def get_discriminating_evidence(self, candidate_conditions, scores, asked):
        best_e, best_gain = None, -1.0

        max_s = max(scores[c] for c in candidate_conditions)
        post = {c: math.exp(scores[c] - max_s) for c in candidate_conditions}
        Z = sum(post.values())
        post = {c: p / Z for c, p in post.items()}

        for c in candidate_conditions:
            for _, e, _ in self.G.out_edges(c, data=True):
                if self.G.nodes[e]["type"] != "evidence" or e in asked:
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

    # ---------------- MAIN TEST-TIME TRAVERSAL ----------------
    def run(
        self,
        scores,
        evidences,
        pathology,
        k=5,
        max_steps=5,
        initial_asked=None
    ):
        asked = set() if initial_asked is None else set(initial_asked)
        observed_yes =  set()
        observed_no = set()

        # Initialize observed YES from DDXPlus evidences
        for ev in evidences:
            if "_@_" in ev:
                observed_yes.add(ev.split("_@_")[0])
            else:
                observed_yes.add(ev)

        # Traversal loop
        for _ in range(max_steps):
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            candidate_conditions = [c for c, _ in ranked[:10]]

            if len(candidate_conditions) <= 1:
                break

            evidence = self.get_discriminating_evidence(
                candidate_conditions, scores, asked
            )
            if evidence is None:
                break

            parent = self.G.nodes[evidence].get("parent")

            # ---------- Parent gating ----------
            if parent and parent != evidence:
                if parent not in observed_yes and parent not in asked:
                    is_yes = parent in observed_yes
                    for c in scores:
                        p = (
                            self.G.edges[c, parent]["p_e_given_c"]
                            if self.G.has_edge(c, parent)
                            else self.SMOOTH
                        )
                        if is_yes:
                            scores[c] = self.capped_add(scores[c], self.safe_log(p))
                        else:
                            if p >= self.ABSENCE_PROB_THRESHOLD:
                                scores[c] = self.capped_add(
                                    scores[c],
                                    self.ABSENCE_WEIGHT * self.safe_log(1 - p)
                                )
                    asked.add(parent)

                if parent in observed_no:
                    continue

            asked.add(evidence)

            # ---------------- Binary evidence ----------------
            values = [
                v for _, v in self.G.out_edges(evidence)
                if self.G.nodes[v]["type"].startswith("possible")
            ]

            if not values:
                is_yes = evidence in observed_yes
                for c in scores:
                    p = (
                        self.G.edges[c, evidence]["p_e_given_c"]
                        if self.G.has_edge(c, evidence)
                        else self.SMOOTH
                    )
                    if is_yes:
                        scores[c] = self.capped_add(scores[c], self.safe_log(p))
                    else:
                        if p >= self.ABSENCE_PROB_THRESHOLD:
                            scores[c] = self.capped_add(
                                scores[c],
                                self.ABSENCE_WEIGHT * self.safe_log(1 - p)
                            )
                continue

            # ---------------- Value-based evidence ----------------
            shown_values = self.get_top_values(evidence, candidate_conditions)
            chosen_values = self.find_existing_values(
                evidences, shown_values, evidence
            )

            if not chosen_values:
                observed_no.add(evidence)
                for c in scores:
                    p = (
                        self.G.edges[c, evidence]["p_e_given_c"]
                        if self.G.has_edge(c, evidence)
                        else self.SMOOTH
                    )
                    if p >= self.ABSENCE_PROB_THRESHOLD:
                        scores[c] = self.capped_add(
                            scores[c],
                            self.ABSENCE_WEIGHT * self.safe_log(1 - p)
                        )
                continue

            observed_yes.add(evidence)
            for c in scores:
                best_pv = self.SMOOTH
                for v in chosen_values:
                    stats = self.G.edges[evidence, v].get("cond_stats", {})
                    best_pv = max(
                        best_pv,
                        stats.get(c, {}).get("p_v_given_e_c", self.SMOOTH)
                    )
                scores[c] = self.capped_add(scores[c], self.safe_log(best_pv))

        # ---------------- Evaluation ----------------
        final_ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_k = {c for c, _ in final_ranked[:k]}
        return pathology in top_k
    