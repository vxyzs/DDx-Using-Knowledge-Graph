import ast
import json
import pickle
import pandas as pd
import random
from collections import defaultdict
import os

from core.test_only_traversal import TestOnlyTraversal
from core.traversal import KG_Traversal
from core.nlu import DDxGraphNLU
from core.parser import Parser

DATA_PATH = "./Data/ddxplus/release_test_patients.csv"
N_SAMPLES = 10000
RANDOM_SEED = 42
TOP_KS = [1, 3, 5]
MAX_STEPS = 7
PARTIAL_RATIO = 0.50

print("Loading KG...")
G = pickle.load(open("./Pickle/kg.pkl", "rb"))

with open("./Data/ddxplus/release_evidences.json", "r") as f:
    evidences_file = json.load(f)

print("Loading test patients...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} patients.")

tester = TestOnlyTraversal(G)
nlu = DDxGraphNLU(G)
parser = Parser()

conditions = [c for c in G.nodes if G.nodes[c]["type"] == "condition"]

def parse_full_evidences(row):
    """
    Extract and structure full evidence strings from patient record rows.

    Args:
        row (pandas.Series): Row from patient dataset.

    Returns:
        list: Structured evidence strings.
    """
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

def sample_partial_evidences(all_evidences, pathology, G):
    """
    Filter and sample a partial subset of patient evidences that are highly likely for the pathology.

    Args:
        all_evidences (list): Complete list of true patient evidences.
        pathology (str): Pathology condition ID.
        G (networkx.Graph): Knowledge graph database.

    Returns:
        list: Sampled subset of evidences.
    """
    k = max(1, int(len(all_evidences) * PARTIAL_RATIO))
    scored_evidences = []
    for ev in all_evidences:
        if "_@_" in ev:
            eid, vid = ev.split("_@_")
            stats = G.edges[eid, vid].get("cond_stats", {}) if G.has_edge(eid, vid) else {}
            p = stats.get(pathology, {}).get("p_v_given_e_c", 1e-6)
        else:
            p = G.edges[pathology, ev]["p_e_given_c"] if G.has_edge(pathology, ev) else 1e-6
        scored_evidences.append((ev, p))
        
    scored_evidences.sort(key=lambda x: x[1], reverse=True)
    return [ev for ev, p in scored_evidences[:k]]

def parse_evidence_format(evid_list):
    """
    Convert dataset evidences list to list and values format expected by traversal algorithms.

    Args:
        evid_list (list): Evidences list.

    Returns:
        Tuple[list, list]: Tuple containing:
            - List of evidence keys.
            - List of corresponding values ('YES' or nested lists).
    """
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
    """
    Determine the numerical rank of the true pathology within current scores.

    Args:
        scores (dict): Dictionary mapping conditions to likelihood probabilities.
        pathology (str): True condition ID.

    Returns:
        int: Rank position (1-indexed).
    """
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for i, (c, _) in enumerate(ranked, start=1):
        if c == pathology:
            return i
    return len(scores)

def evaluate_scenario(df_subset, scenario_name, use_partial=False):
    """
    Evaluate traversal accuracy, steps, and probabilities on a dataset subset under a given scenario.

    Args:
        df_subset (pandas.DataFrame): Patient dataset subset to run.
        scenario_name (str): Label identifier for the current run configuration.
        use_partial (bool): True to use partial/masked evidences, False for full info.

    Returns:
        dict: Performance and metrics dictionary mapping results.
    """
    print(f"\n=== Evaluating: {scenario_name} ===")

    results = {
        "topk_hits": {k: 0 for k in TOP_KS},
        "mrr": 0.0,
        "mean_rank": 0.0,
        "gt_scores": [],
        "top1_scores": [],
        "score_gaps": [],
        "rank_histogram": defaultdict(int),
        "gt_prob_thresholds": {0.5: 0, 0.6: 0, 0.7: 0},
        "steps": 0,
        "total": 0,
    }

    for _, row in df_subset.iterrows():
        pathology = row["PATHOLOGY"]
        all_evidences = parse_full_evidences(row)

        if use_partial:
            evid_list = sample_partial_evidences(all_evidences, pathology, G)
        else:
            evid_list = all_evidences

        evidences, values = parse_evidence_format(evid_list)
        scores = {c: 0.0 for c in conditions}
        traversal = KG_Traversal(G, scores, nlu, parser)
        traversal.apply_initial_evidence(evidences, values)

        steps_taken = tester.run(
            scores=traversal.scores,
            evidences=evid_list,
            pathology=pathology,
            k=max(TOP_KS),
            max_steps=MAX_STEPS,
            initial_asked=traversal.asked,
        )

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

        for t in [0.5, 0.6, 0.7]:
            if gt_score >= t:
                results["gt_prob_thresholds"][t] += 1

        for k in TOP_KS:
            if rank <= k:
                results["topk_hits"][k] += 1

        results["total"] += 1

    total = results["total"]
    if total == 0:
        print("No samples to evaluate.")
        return results

    print("\n--- RESULTS ---")
    for k in TOP_KS:
        acc = results["topk_hits"][k] / total
        print(f"Top-{k} Accuracy: {acc:.4f}")

    mrr = results["mrr"] / total
    mean_rank = results["mean_rank"] / total
    avg_steps = results["steps"] / total

    print(f"MRR: {mrr:.4f}")
    print(f"Mean Rank: {mean_rank:.4f}")

    avg_gt_score = sum(results["gt_scores"]) / total
    avg_top1_score = sum(results["top1_scores"]) / total
    avg_score_gap = sum(results["score_gaps"]) / total

    print("\n--- PROBABILITY ANALYSIS ---")
    print(f"Avg GT Prob: {avg_gt_score:.4f}")
    print(f"Avg Top-1 Prob: {avg_top1_score:.4f}")
    print(f"Avg Prob Gap: {avg_score_gap:.4f}")

    print("\n--- CONFIDENCE METRICS ---")
    for t in [0.5, 0.6, 0.7]:
        acc_at_t = results["gt_prob_thresholds"][t] / total
        print(f"GT Prob >= {t}: {acc_at_t:.4f}")
        
    results["gt_prob_thresholds"] = {t: results["gt_prob_thresholds"][t] / total for t in [0.5, 0.6, 0.7]}

    print("\n--- SYSTEM ---")
    print(f"Avg Steps: {avg_steps:.2f}")

    print("\n--- RANK DISTRIBUTION ---")
    for r in sorted(results["rank_histogram"].keys()):
        print(f"Rank {r}: {results['rank_histogram'][r]}")

    results["mrr"] = mrr
    results["mean_rank"] = mean_rank
    results["steps"] = avg_steps
    results["topk_accuracy"] = {
        k: results["topk_hits"][k] / total for k in TOP_KS
    }

    return results

def filter_difficult_cases(df):
    """
    Extract subset of rows considered difficult based on the diagnosis ranks.

    Args:
        df (pandas.DataFrame): Patient dataset.

    Returns:
        pandas.DataFrame: Hard cases dataframe.
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

def evaluate_per_pathology_and_save(df, output_path="./results/per_pathology_results.json"):
    """
    Evaluate traversal systems pathology by pathology under multiple information configs.

    Args:
        df (pandas.DataFrame): Full patient dataset.
        output_path (str): File destination path for JSON reports.
    """
    all_results = {}
    unique_pathologies = df["PATHOLOGY"].unique()
    print(f"\nFound {len(unique_pathologies)} unique pathologies")

    for pathology_name in unique_pathologies:
        print(f"\n==============================")
        print(f"Processing pathology: {pathology_name}")
        print(f"==============================")

        df_path = df[df["PATHOLOGY"] == pathology_name]
        sample_size = min(1000, len(df_path))
        df_sample = df_path.sample(n=sample_size, random_state=RANDOM_SEED)

        pathology_results = {}

        full_results = evaluate_scenario(
            df_sample,
            scenario_name=f"{pathology_name} - Full Info",
            use_partial=False
        )
        full_results["rank_histogram"] = dict(full_results["rank_histogram"])
        pathology_results["full_info"] = full_results

        partial_results = evaluate_scenario(
            df_sample,
            scenario_name=f"{pathology_name} - Partial Info",
            use_partial=True
        )
        partial_results["rank_histogram"] = dict(partial_results["rank_histogram"])
        pathology_results["partial_info"] = partial_results

        hard_df = filter_difficult_cases(df_sample)
        if len(hard_df) > 0:
            print(f"\nHard cases found: {len(hard_df)}")
            hard_results = evaluate_scenario(
                hard_df,
                scenario_name=f"{pathology_name} - Hard Cases",
                use_partial=True
            )
            hard_results["rank_histogram"] = dict(hard_results["rank_histogram"])
            pathology_results["hard_cases"] = {
                "num_samples": len(hard_df),
                "metrics": hard_results
            }
        else:
            print("No hard cases found.")
            pathology_results["hard_cases"] = {
                "num_samples": 0,
                "metrics": None
            }

        all_results[pathology_name] = {
            "num_samples": sample_size,
            "scenarios": pathology_results
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSaved results to {output_path}")

if __name__ == "__main__":
    random.seed(RANDOM_SEED)

    # 1. FULL INFO
    # evaluate_scenario(df, "Full Information", use_partial=False)

    # 2. PARTIAL INFO
    # evaluate_scenario(df, "Partial Information (50%)", use_partial=True)

    # # 3. HARD CASES
    # hard_df = filter_difficult_cases(df)

    # if len(hard_df) > 0:
    #     evaluate_scenario(hard_df, "Difficult Cases", use_partial=True)
    # else:
    #     print("\nNo difficult cases found in sample.")

    # 4. PER-PATHOLOGY JSON GENERATION
    evaluate_per_pathology_and_save(df)