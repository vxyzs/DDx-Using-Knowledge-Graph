import re

from core.traversal import BaseTraversal


class TestOnlyTraversal(BaseTraversal):
    """
    Simulation traversal engine designed to evaluate diagnostic traversal
    performance against ground truth dataset values.
    """

    VALUE_PATTERN = re.compile(r'^(E_\d+)_@_(V_\d+|\d+)$')

    def __init__(self, G):
        """
        Initialize simulation traversal engine.

        Args:
            G (networkx.Graph): The disease-evidence knowledge graph.
        """
        super().__init__(G)

    def get_top_values(self, evidence, candidate_conditions):
        """
        Retrieve highest scoring categorical values for the specified evidence
        across candidate conditions.
        """
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
        """
        Filter matched values from full evidence list that correspond to
        specified evidence ID.
        """
        chosen = []
        for ev in evidences:
            m = self.VALUE_PATTERN.match(ev)
            if not m:
                continue
            eid, vid = m.group(1), m.group(2)
            if eid == evidence and vid in shown_values:
                chosen.append(ev)
        return chosen

    def get_discriminating_evidence(
        self, candidate_conditions, scores, asked
    ):
        """
        Locate next informative evidence using base class solver.
        """
        return self._compute_discriminating_evidence(
            candidate_conditions, scores, asked
        )

    def run(
        self,
        scores,
        evidences,
        pathology,
        k=5,
        max_steps=5,
        initial_asked=None
    ):
        """
        Run the simulation diagnostic traversal over patient data.

        Args:
            scores (dict): Dictionary mapping conditions to current likelihood
              scores.
            evidences (list): Complete list of true patient symptoms.
            pathology (str): The true ground-truth disease pathology.
            k (int): Top k candidate conditions to output.
            max_steps (int): Limit on traversal step count.
            initial_asked (set, optional): Initial set of queried evidence keys.

        Returns:
            int: Number of traversal steps taken.
        """
        asked = set() if initial_asked is None else set(initial_asked)
        observed_yes = set()
        observed_no = set()

        for ev in evidences:
            if "_@_" in ev:
                observed_yes.add(ev.split("_@_")[0])
            else:
                observed_yes.add(ev)

        steps_taken = 0
        for _ in range(max_steps):
            steps_taken += 1
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
                            scores[c] = self.capped_add(
                                scores[c], self.safe_log(p)
                            )
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
                        scores[c] = self.capped_add(
                            scores[c], self.safe_log(p)
                        )
                    else:
                        if p >= self.ABSENCE_PROB_THRESHOLD:
                            scores[c] = self.capped_add(
                                scores[c],
                                self.ABSENCE_WEIGHT * self.safe_log(1 - p)
                            )
                continue

            shown_values = self.get_top_values(
                evidence, candidate_conditions
            )
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
                    stats = self.G.edges[evidence, v].get(
                        "cond_stats", {}
                    )
                    best_pv = max(
                        best_pv,
                        stats.get(c, {}).get(
                            "p_v_given_e_c", self.SMOOTH
                        )
                    )
                scores[c] = self.capped_add(
                    scores[c], self.safe_log(best_pv)
                )

        self._convert_scores_to_probabilities(scores)

        return steps_taken