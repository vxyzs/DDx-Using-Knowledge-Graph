import pickle
import pandas as pd
import ast
import random

from traversal import KG_Traversal


# =========================
# LOAD DATA
# =========================
G = pickle.load(open("../Pickle/kg.pkl", "rb"))
cond_count = pickle.load(open("../Pickle/cond_count.pkl", "rb"))

df = pd.read_csv(".././Data/ddxplus/release_test_patients.csv")


# =========================
# PARSE PATIENT (CLEAN)
# =========================
def parse_patient(row):
    evidences_raw = set()

    evidences_raw.add(row["INITIAL_EVIDENCE"])
    evidences_raw.update(ast.literal_eval(row["EVIDENCES"]))

    patient_map = {}

    for ev in evidences_raw:
        if "_@_" in ev:
            eid, vid = ev.split("_@_")

            full_val = f"{eid}_@_{vid}"

            if eid not in patient_map:
                patient_map[eid] = []

            patient_map[eid].append(full_val)

        else:
            patient_map[ev] = "YES"

    return patient_map


# =========================
# SIMULATE ANSWER
# =========================
def simulate_answer(G, evidence, patient_map):

    if evidence in patient_map:
        val = patient_map[evidence]

        if isinstance(val, list):
            return val

        return "YES"

    return "NO"


# =========================
# INITIALIZE SCORES
# =========================
def init_scores():
    return {c: 0.0 for c in cond_count}


# =========================
# RUN SINGLE CASE
# =========================
def run_case(G, patient_map, true_pathology, init_evidences, max_steps=10):

    scores = init_scores()
    traversal = KG_Traversal(G, scores)

    # ---------- INITIAL EVIDENCE ----------
    evidences = []
    values = []

    for e in init_evidences:

        if e in patient_map:
            val = patient_map[e]

            if isinstance(val, list):
                evidences.append(e)
                values.append(val)
            else:
                evidences.append(e)
                values.append("YES")
        else:
            evidences.append(e)
            values.append("NO")

    traversal.apply_initial_evidence(evidences, values)

    steps = 0

    # ---------- INTERACTIVE LOOP ----------
    for _ in range(max_steps):

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        candidate_conditions = [c for c, _ in ranked[:10]]

        if ranked[0][0] == true_pathology:
            break

        evidence = traversal.get_discriminating_evidence(candidate_conditions)
        if evidence is None:
            break

        answer = simulate_answer(G, evidence, patient_map)

        if isinstance(answer, list):
            traversal.apply_value_answer(evidence, answer)
        else:
            traversal.apply_binary_answer(evidence, answer == "YES")

        steps += 1

    # ---------- FINAL RANK ----------
    final_ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    rank = 1000
    for i, (c, _) in enumerate(final_ranked):
        if c == true_pathology:
            rank = i + 1
            break

    return rank, steps


# =========================
# EVALUATE
# =========================
def evaluate(df, mode="full", sample_size=2000):

    results = {
        "top1": 0,
        "top3": 0,
        "top5": 0,
        "mrr": 0.0,
        "steps": 0,
        "total": 0
    }

    df_sample = df.sample(n=sample_size, random_state=42)

    for _, row in df_sample.iterrows():

        true_pathology = row["PATHOLOGY"]
        patient_map = parse_patient(row)

        evid_list = list(patient_map.keys())

        # ---------- MODES ----------
        if mode == "full":
            init_evidences = evid_list

        elif mode == "partial":
            k = max(1, len(evid_list) // 2)
            init_evidences = random.sample(evid_list, k)

        elif mode == "hard":
            if len(evid_list) < 8:
                continue
            k = len(evid_list) // 2
            init_evidences = random.sample(evid_list, k)

        else:
            raise ValueError("Invalid mode")

        # ---------- RUN ----------
        rank, steps = run_case(
            G,
            patient_map,
            true_pathology,
            init_evidences
        )

        # ---------- METRICS ----------
        results["total"] += 1

        if rank == 1:
            results["top1"] += 1
        if rank <= 3:
            results["top3"] += 1
        if rank <= 5:
            results["top5"] += 1

        results["mrr"] += 1.0 / rank
        results["steps"] += steps

    total = results["total"]

    return {
        "Top-1": results["top1"] / total,
        "Top-3": results["top3"] / total,
        "Top-5": results["top5"] / total,
        "MRR": results["mrr"] / total,
        "Avg Steps": results["steps"] / total
    }


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    for mode in ["full", "partial", "hard"]:
        print(f"\n=== {mode.upper()} ===")

        res = evaluate(df, mode=mode, sample_size=2000)

        for k, v in res.items():
            print(f"{k}: {v:.4f}")