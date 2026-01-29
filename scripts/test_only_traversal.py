import ast
import pickle
import pandas as pd
from collections import defaultdict

from core.test_only_traversal import TestOnlyTraversal
from core.traversal import KG_Traversal 

DATA_PATH = "./Data/ddxplus/release_test_patients.csv"

N_SAMPLES = 100
RANDOM_SEED = 42

TOP_KS = [1, 3, 5, 7]
MAX_STEPS_LIST = [4, 6, 8, 10]


G = pickle.load(open("./Pickle/kg.pkl", "rb"))

print("Loading test patients...")
df = pd.read_csv(DATA_PATH)
df = df.sample(n=N_SAMPLES, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"Loaded {len(df)} test patients.")


# ---------------- CONDITION LIST ----------------
cond_count = defaultdict(int)
for _, row in df.iterrows():
    cond_count[row["PATHOLOGY"]] += 1

conditions = list(cond_count.keys())


# ---------------- INITIAL EVIDENCE PARSER ----------------
def parse_initial_evidence(init_evid):
    """
    Parses INITIAL_EVIDENCE field into
    - evidences: list[str]
    - values: list[str | list[str] | None]
    """
    if "_@_" in init_evid:
        eid, vid = init_evid.split("_@_")
        return [eid], [[vid]]
    else:
        return [init_evid], ["YES"]


# ---------------- EVALUATION ----------------
tester = TestOnlyTraversal(G)

accuracies = {}

for max_steps in MAX_STEPS_LIST:
    for k in TOP_KS:
        total_hits = 0
        total_tests = 0

        print(f"\nEvaluating: top-{k}, max_steps={max_steps}")
        traversal_init = KG_Traversal(G, scores)

        for _, row in df.iterrows():
            pathology = row["PATHOLOGY"]

            # Fresh scores per patient (MANDATORY)
            scores = {c: 0.0 for c in conditions}

            # ---------- Initial evidence ----------
            init_evid = row["INITIAL_EVIDENCE"]
            evidences, values = parse_initial_evidence(init_evid)

            # Use KG_Traversal only to apply initial evidence cleanly
            
            traversal_init.apply_initial_evidence(evidences, values)

            initial_asked = traversal_init.asked

            # ---------- Full evidence list ----------
            full_evidences = [init_evid] + ast.literal_eval(row["EVIDENCES"])

            # ---------- Run test-only traversal ----------
            hit = tester.run(
                scores=scores,
                evidences=full_evidences,
                pathology=pathology,
                k=k,
                max_steps=max_steps,
                initial_asked=initial_asked
            )

            total_tests += 1
            if hit:
                total_hits += 1

        accuracy = total_hits / total_tests
        accuracies[(k, max_steps)] = accuracy

        print(f"Top-{k} accuracy @ max_steps={max_steps}: {accuracy:.4f}")


# ---------------- SUMMARY ----------------
print("\n=== FINAL ACCURACY SUMMARY ===")
for (k, steps), acc in sorted(accuracies.items()):
    print(f"Top-{k} | max_steps={steps} → {acc:.4f}")
