import pickle
from functools import lru_cache

from core.nlu import DDxGraphNLU
from core.parser import Parser
from core.traversal import KG_Traversal


@lru_cache(maxsize=1)
def load_graph(path="./Pickle/kg.pkl"):
    """
    Load the serialized disease-symptom knowledge graph.

    Args:
        path (str): Relative file path to the pickle graph representation.

    Returns:
        networkx.Graph: The loaded graph.
    """
    with open(path, "rb") as f:
        G = pickle.load(f)
    return G


def initialize_scores(G):
    """
    Initialize score records for all condition nodes in the knowledge graph.

    Args:
        G (networkx.Graph): Knowledge graph database.

    Returns:
        dict: Dictionary mapping condition names/IDs to a start score of 0.0.
    """
    condition_nodes = [
        node
        for node, data in G.nodes(data=True)
        if data.get("type") == "condition"
    ]
    return dict.fromkeys(condition_nodes, 0.0)


def main():
    """
    Main execution routine for running the interactive DDx diagnosis console
    script.
    """
    try:
        print("Loading knowledge graph...")
        G = load_graph()
    except FileNotFoundError:
        print(
            "Knowledge Graph pickle file not found. "
            "Ensure './Pickle/kg.pkl' exists."
        )
        return

    scores = initialize_scores(G)
    user_input = input("Describe your symptoms: ")

    nlu = DDxGraphNLU(G)
    parser = Parser()

    traversal = KG_Traversal(
        G,
        scores,
        nlu,
        parser,
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