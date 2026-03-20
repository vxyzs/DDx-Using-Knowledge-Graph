import ast
import pickle
import pandas as pd
import random
from collections import defaultdict

from core.test_only_traversal import TestOnlyTraversal
from core.traversal import KG_Traversal
from core.nlu import DDxGraphNLU
from core.parser import Parser

# ---------------- CONFIG ----------------

DATA_PATH = "./Data/ddxplus/release_test_patients.csv"

N_SAMPLES = 100
RANDOM_SEED = 42

TOP_KS = [1, 3, 5]
MAX_STEPS = 10

PARTIAL_RATIO = 0.5  # 50% evidences

# ---------------- LOAD ----------------

print("Loading KG...")
G = pickle.load(open("./Pickle/kg.pkl", "rb"))

print("Loading test patients...")
df = pd.read_csv(DATA_PATH)
df = df.sample(n=N_SAMPLES, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"Loaded {len(df)} patients.")

# ---------------- INIT ----------------

tester = TestOnlyTraversal(G)
nlu = DDxGraphNLU(G)
parser = Parser()

conditions = [c for c in G.nodes if G.nodes[c]["type"] == "condition"]

# ---------------- HELPERS ----------------


def parse_full_evidences(row):
    """Return all evidences (full info scenario)"""
    return [row["INITIAL_EVIDENCE"]] + ast.literal_eval(row["EVIDENCES"])


def sample_partial_evidences(all_evidences):
    """Random 50% sampling"""
    k = max(1, int(len(all_evidences) * PARTIAL_RATIO))
    return random.sample(all_evidences, k)


def parse_evidence_format(evid_list):
    """Convert dataset format → traversal format"""
    evidences, values = [], []

    for ev in evid_list:
        if "_@_" in ev:
            eid, vid = ev.split("_@_")
            evidences.append(eid)
            values.append([vid])
        else:
            evidences.append(ev)
            values.append("YES")

    return evidences, values


def get_rank(scores, pathology):
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for i, (c, _) in enumerate(ranked, start=1):
        if c == pathology:
            return i
    return len(scores)


# ---------------- EVALUATION ----------------


def evaluate_scenario(df_subset, scenario_name, use_partial=False):
    print(f"\n=== Evaluating: {scenario_name} ===")

    results = {
        "topk_hits": {k: 0 for k in TOP_KS},
        "mrr": 0.0,
        "steps": 0,
        "total": 0,
    }

    for _, row in df_subset.iterrows():
        pathology = row["PATHOLOGY"]

        # -------- Prepare evidences --------
        all_evidences = parse_full_evidences(row)

        if use_partial:
            evid_list = sample_partial_evidences(all_evidences)
        else:
            evid_list = all_evidences

        evidences, values = parse_evidence_format(evid_list)

        # -------- Initialize scores --------
        scores = {c: 0.0 for c in conditions}
        traversal = KG_Traversal(G, scores)

        # Apply initial evidences
        traversal.apply_initial_evidence(evidences, values)

        # -------- Run traversal --------
        tester.run(
            scores=traversal.scores,
            evidences=evid_list,
            pathology=pathology,
            k=max(TOP_KS),
            max_steps=MAX_STEPS,
            initial_asked=traversal.asked,
        )

        # -------- Evaluation --------
        rank = get_rank(traversal.scores, pathology)

        results["mrr"] += 1.0 / rank
        results["steps"] += MAX_STEPS  # approximation

        for k in TOP_KS:
            if rank <= k:
                results["topk_hits"][k] += 1

        results["total"] += 1

    # -------- Aggregate --------
    total = results["total"]

    print("\n--- RESULTS ---")
    for k in TOP_KS:
        acc = results["topk_hits"][k] / total
        print(f"Top-{k} Accuracy: {acc:.4f}")

    print(f"MRR: {results['mrr'] / total:.4f}")
    print(f"Avg Steps: {results['steps'] / total:.2f}")

    return results


# ---------------- DIFFICULT CASE FILTER ----------------


def filter_difficult_cases(df):
    """
    Difficult = true disease rank > 5 in dataset differential list
    """
    hard_cases = []

    for _, row in df.iterrows():
        try:
            ddx_list = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
            if row["PATHOLOGY"] in ddx_list:
                rank = ddx_list.index(row["PATHOLOGY"]) + 1
                if rank > 5:
                    hard_cases.append(row)
        except Exception:
            continue

    return pd.DataFrame(hard_cases)


# ---------------- RUN ALL SCENARIOS ----------------

if __name__ == "__main__":
    random.seed(RANDOM_SEED)

    # 1. FULL INFO
    evaluate_scenario(df, "Full Information", use_partial=False)

    # 2. PARTIAL INFO
    evaluate_scenario(df, "Partial Information (50%)", use_partial=True)

    # 3. HARD CASES
    hard_df = filter_difficult_cases(df)

    if len(hard_df) > 0:
        evaluate_scenario(hard_df, "Difficult Cases", use_partial=True)
    else:
        print("\nNo difficult cases found in sample.")