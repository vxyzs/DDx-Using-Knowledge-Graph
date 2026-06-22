import math
from abc import ABC, abstractmethod

from core.config import load_config
from core.logger import get_logger
from core.nlu import DDxGraphNLU
from core.parser import Parser

logger = get_logger("traversal")

config = load_config()
traversal_config = config["traversal"]


class BaseTraversal(ABC):
    """
    Abstract base class representing generic Diagnostic Traversal engines
    over the DDx Knowledge Graph.
    """
    SMOOTH = traversal_config["smooth"]
    MAX_DELTA = traversal_config["max_delta"]
    ABSENCE_PROB_THRESHOLD = traversal_config["absence_prob_threshold"]
    ABSENCE_WEIGHT = traversal_config["absence_weight"]

    def __init__(self, G):
        """
        Initialize the traversal engine base with the knowledge graph.

        Args:
            G (networkx.Graph): The disease-evidence knowledge graph.
        """
        self.G = G

    def safe_log(self, p: float) -> float:
        """
        Calculate log probability safely preventing log(0) error.
        """
        return math.log(max(p, self.SMOOTH))

    def capped_add(self, score: float, delta: float) -> float:
        """
        Apply bound capping on score changes to prevent floating point overflow
        or underflow.
        """
        return score + max(-self.MAX_DELTA, min(self.MAX_DELTA, delta))

    def _compute_discriminating_evidence(
        self, candidate_conditions, scores, asked
    ):
        """
        Identify the most informative symptom node to query next across
        candidate conditions.

        Args:
            candidate_conditions (list): List of candidate disease codes.
            scores (dict): Dictionary mapping conditions to their current
              likelihood scores.
            asked (set): Set of evidence IDs already queried.

        Returns:
            str or None: The selected evidence ID, or None if none remain.
        """
        best_e, best_gain = None, -1.0

        T=2.0
        max_s = max(scores[c] for c in candidate_conditions)
        post = {c: math.exp((scores[c] - max_s) / T) for c in candidate_conditions}
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

    def _convert_scores_to_probabilities(self, scores):
        """
        Convert raw accumulative disease score values to normalized
        probability distributions.

        Args:
            scores (dict): Dictionary of condition scores to normalize.
        """
        if not scores:
            return
        max_score = max(scores.values())
        exp_scores = {
            c: math.exp(s - max_score) for c, s in scores.items()
        }
        sum_exp = sum(exp_scores.values())
        if sum_exp > 0:
            for c in scores:
                scores[c] = exp_scores[c] / sum_exp
        else:
            for c in scores:
                scores[c] = 0.0

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Run the interactive or simulation traversal process.
        """
        pass


class KG_Traversal(BaseTraversal):
    """
    Interactive clinical traversal engine that prompts the doctor/patient
    for details and refines differential diagnosis candidate conditions
    recursively.
    """

    def __init__(self, G, scores, nlu, parser, user_input=None):
        """
        Initialize interactive diagnostic traversal.

        Args:
            G (networkx.Graph): The knowledge graph.
            scores (dict): Initial scores dictionary.
            nlu (DDxGraphNLU): Natural language understanding retriever.
            parser (Parser): LLM-based symptom value parser.
            user_input (str, optional): Raw user initial complaint text.
        """
        super().__init__(G)
        self.nlu = nlu
        self.parser = parser
        self.scores = scores

        self.asked = set()
        self.observed_yes = set()
        self.observed_no = set()
        self.evidence_answers = {}

        if user_input:
            self._parse_initial_query(user_input)

    def _parse_initial_query(self, user_input):
        """
        Parse free-text user input, extract initial evidences, and compute
        initial scores.
        """
        context = self.nlu.retrieve(user_input)
        evidences, values = self.parser.parse_query(user_input, context)

        if not evidences:
            return

        logger.info("Parsed initial evidence:")
        for e, v in zip(evidences, values):
            logger.info(f"  {e} → {v}")

        self.apply_initial_evidence(evidences=evidences, values=values)

    def _format_answer(self, val):
        if val in ("YES", "NO"):
            return val
        if isinstance(val, list):
            res = []
            for v in val:
                if self.G.has_node(v):
                    res.append(self.G.nodes[v].get("value_en", v))
                else:
                    res.append(v)
            return ", ".join(res)
        return str(val)

    def get_discriminating_evidence(self, candidate_conditions):
        """
        Locate next informative evidence using the base class solver.
        """
        return self._compute_discriminating_evidence(
            candidate_conditions, self.scores, self.asked
        )

    def apply_binary_answer(self, evidence, is_yes):
        """
        Apply score adjustment based on yes/no symptom answer.
        """
        self.evidence_answers[evidence] = "YES" if is_yes else "NO"
        for c in self.scores:
            p = (
                self.G.edges[c, evidence]["p_e_given_c"]
                if self.G.has_edge(c, evidence)
                else self.SMOOTH
            )

            if is_yes:
                self.scores[c] = self.capped_add(
                    self.scores[c], self.safe_log(p)
                )
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
        """
        Apply score adjustment based on selected/parsed categorical symptom
        values.
        """
        self.observed_yes.add(evidence)
        self.evidence_answers[evidence] = self._format_answer(chosen_values)
        for c in self.scores:
            best_pv = self.SMOOTH
            for v in chosen_values:
                stats = self.G.edges[evidence, v].get("cond_stats", {})
                best_pv = max(
                    best_pv,
                    stats.get(c, {}).get("p_v_given_e_c", self.SMOOTH)
                )

            self.scores[c] = self.capped_add(
                self.scores[c], self.safe_log(best_pv)
            )

        self.asked.add(evidence)

    def apply_initial_evidence(self, evidences, values):
        """
        Apply batch initial evidence list to initialize scoring state.
        """
        for evid, val in zip(evidences, values):
            self.asked.add(evid)
            self.evidence_answers[evid] = self._format_answer(val)

            parent = self.G.nodes[evid].get("parent")
            if val not in ("YES", "NO") and parent:
                self.asked.add(parent)

            if val == "YES":
                self.observed_yes.add(evid)
            elif val == "NO":
                self.observed_no.add(evid)

            if val not in ("YES", "NO"):
                for v in val:
                    for c in self.scores:
                        stats = self.G.edges[evid, v].get("cond_stats", {})
                        p = stats.get(c, {}).get("p_v_given_e_c", self.SMOOTH)
                        self.scores[c] = self.capped_add(self.scores[c], self.safe_log(p))
            else:
                self.apply_binary_answer(evid, val == "YES")

    def run(self, max_steps=5, top_k_conditions=10):
        """
        Run the interactive clinical diagnostic loop.
        """
        logger.info("=== INTERACTIVE DIAGNOSTIC TRAVERSAL ===")

        for step in range(max_steps):
            T = 2.0
            max_s = max(self.scores.values()) if self.scores else 0.0
            exp_scores = {c: math.exp((s - max_s) / T) for c, s in self.scores.items()}
            sum_exp = sum(exp_scores.values())
            probs = {c: (e / sum_exp) for c, e in exp_scores.items()} if sum_exp > 0 else {c: 0.0 for c in self.scores}

            ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)[
                :top_k_conditions
            ]
            candidate_conditions = [c for c, _ in ranked]

            logger.info(f"Step {step + 1} — Current Differential:")
            for c, p in ranked:
                logger.info(f"  {c:40s} prob={p*100:.2f}% (score={self.scores[c]:.4f})")

            if len(candidate_conditions) <= 1:
                break

            evidence = self.get_discriminating_evidence(candidate_conditions)
            if evidence is None:
                break

            parent = self.G.nodes[evidence].get("parent", None)
            logger.info(f"Parent evidence: {parent}")

            if (
                parent
                and parent != evidence
                and parent not in self.observed_yes
            ):
                if parent not in self.asked:
                    question = self.G.nodes[parent]["question_en"]
                    print(f"\n🩺 Question: {question}")
                    ans = input("Your answer: ").strip()

                    context_text = (
                        f"The doctor asked: '{question}'. "
                        f"The patient answered: '{ans}'"
                    )
                    context = self.nlu.retrieve(context_text)
                    ext_evidences, ext_values = self.parser.parse_query(
                        context_text, context
                    )

                    if ext_evidences:
                        self.apply_initial_evidence(
                            ext_evidences, ext_values
                        )

                    self.asked.add(parent)

                if parent in self.observed_no:
                    continue

            self.asked.add(evidence)
            question = self.G.nodes[evidence]["question_en"]

            print(f"\n🩺 Question: {question}")
            ans = input("Your answer: ").strip()

            context_text = (
                f"The doctor asked: '{question}'. "
                f"The patient answered: '{ans}'"
            )
            context = self.nlu.retrieve(context_text)
            ext_evidences, ext_values = self.parser.parse_query(
                context_text, context
            )

            if ext_evidences:
                self.apply_initial_evidence(ext_evidences, ext_values)
            step += 1

        logger.info("=== FINAL RANKED CONDITIONS ===")
        results = self.get_diagnostic_results(top_k_conditions=top_k_conditions)
        return results
    
    def get_diagnostic_results(self, top_k_conditions=10):

        top_candidates = sorted(
            self.scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k_conditions]
        for c, s in top_candidates:
            logger.info(f"{c:40s} prob={s:.4f}")

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