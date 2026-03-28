import ast
import json
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
MAX_STEPS = 5

PARTIAL_RATIO = 0.60  #60% evidences

# ---------------- LOAD ----------------

print("Loading KG...")
G = pickle.load(open("./Pickle/kg.pkl", "rb"))

with open("./Data/ddxplus/release_evidences.json", "r") as f:
    evidences_file = json.load(f)

print("Loading test patients...")
df = pd.read_csv(DATA_PATH)
df = df[df["PATHOLOGY"] == "URTI"]
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
    evidences = [row["INITIAL_EVIDENCE"]] + ast.literal_eval(row["EVIDENCES"])
    for i, ev in enumerate(evidences):
        if "_@_" in ev:
            eid, vid = ev.split("_@_")
            if vid.isdigit():
                vid = int(vid)
                values = evidences_file.get(eid, {}).get("possible-values", []) 
                if vid == 0:
                    vid = values[0]
                elif vid >= 1 and vid <= 3:
                    vid = values[1]
                elif vid >= 4 and vid <= 7:
                    vid = values[2]
                elif vid >= 6 and vid <= 10:
                    vid = values[3]
            evidences[i] = f"{eid}_@_{vid}"
    return evidences


def sample_partial_evidences(all_evidences):
    """Random 50% sampling"""
    k = max(1, int(len(all_evidences) * PARTIAL_RATIO))
    return random.sample(all_evidences, k)


def parse_evidence_format(evid_list):
    """Convert dataset format → traversal format"""
    evidences, values = [], []

    for ev in evid_list:
        if "_@_" in ev:
            eid, _ = ev.split("_@_")
            evidences.append(eid)
            values.append([ev])
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
        "mean_rank": 0.0,
        "gt_scores": [],
        "top1_scores": [],
        "score_gaps": [],
        "rank_histogram": defaultdict(int),
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
        steps_taken = tester.run(
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
        results["mean_rank"] += rank
        results["rank_histogram"][rank] += 1
        results["steps"] += steps_taken

        ranked = sorted(traversal.scores.items(), key=lambda x: x[1], reverse=True)
        top1_score = ranked[0][1] if ranked else 0.0
        gt_score = traversal.scores.get(pathology, 0.0)

        results["gt_scores"].append(gt_score)
        results["top1_scores"].append(top1_score)
        results["score_gaps"].append(top1_score - gt_score)

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
    print(f"Mean Rank: {results['mean_rank'] / total:.4f}")

    avg_gt_score = sum(results["gt_scores"]) / total if total > 0 else 0.0
    avg_top1_score = sum(results["top1_scores"]) / total if total > 0 else 0.0
    avg_score_gap = sum(results["score_gaps"]) / total if total > 0 else 0.0

    print("\n--- SCORE ANALYSIS ---")
    print(f"Avg GT Score: {avg_gt_score:.4f}")
    print(f"Avg Top-1 Score: {avg_top1_score:.4f}")
    print(f"Avg Score Gap: {avg_score_gap:.4f}")

    print("\n--- SYSTEM ---")
    print(f"Avg Steps: {results['steps'] / total:.2f}")

    print("\n--- RANK DISTRIBUTION ---")
    for r in sorted(results["rank_histogram"].keys()):
        print(f"Rank {r}: {results['rank_histogram'][r]}")

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
    # evaluate_scenario(df, "Full Information", use_partial=False)

    # 2. PARTIAL INFO
    evaluate_scenario(df, "Partial Information (50%)", use_partial=True)

    # 3. HARD CASES
    hard_df = filter_difficult_cases(df)

    if len(hard_df) > 0:
        evaluate_scenario(hard_df, "Difficult Cases", use_partial=True)
    else:
        print("\nNo difficult cases found in sample.")