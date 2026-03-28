import pickle
from core.traversal import KG_Traversal

def main():
    G = pickle.load(open("./Pickle/kg.pkl", "rb"))
    # Further processing...

    scores = {c: 0.0 for c in G.nodes if G.nodes[c]["type"] == "condition"}

    user_input = input("Describe your symptoms: ")

    traversal = KG_Traversal(
        G,
        scores,
        user_input=user_input
    )

    result = traversal.run()
    top_conditions_with_scores = result["top_conditions_with_scores"]
    cond_evidence_map = result["cond_evidence_map"]
    missing_evidence_map = result["missing_evidence_map"]

    print("\n=== TOP CONDITIONS WITH SCORES ===")
    for c, s in top_conditions_with_scores.items():
        print(f"{c:40s} score={s:.4f}")
    
    print("\n=== EVIDENCE FOR TOP CONDITIONS ===")
    for c in top_conditions_with_scores:
        evidence = cond_evidence_map.get(c, [])
        print(f"\nCondition: {c}")
        print("Evidence:")
        for e in evidence:
            print(f"  - {e}")
    
    print("\n=== MISSING EVIDENCE FOR TOP CONDITIONS ===")
    for c in top_conditions_with_scores:
        missing_evidence = missing_evidence_map.get(c, [])
        print(f"\nCondition: {c}")
        print("Missing Evidence:")
        for e in missing_evidence:
            print(f"  - {e}")

if __name__ == "__main__":
    main()