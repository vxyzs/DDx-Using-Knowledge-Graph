import pickle
from functools import lru_cache
from core.traversal import KG_Traversal


@lru_cache(maxsize=1)
def load_graph(path="./Pickle/kg.pkl"):
    """
    Loads the knowledge graph once and caches it.
    Prevents re-loading if called multiple times.
    """
    with open(path, "rb") as f:
        G = pickle.load(f)
    return G


def initialize_scores(G):
    """
    Efficiently initialize condition node scores.
    Avoids repeated dictionary lookups.
    """
    condition_nodes = [
        node
        for node, data in G.nodes(data=True)
        if data.get("type") == "condition"
    ]
    return dict.fromkeys(condition_nodes, 0.0)


def main():
    try:
        print("Loading knowledge graph...")
        G = load_graph()
    except FileNotFoundError:
        print("Knowledge Graph pickle file not found. Ensure './Pickle/kg.pkl' exists.")
        return

    scores = initialize_scores(G)

    user_input = input("Describe your symptoms: ")

    traversal = KG_Traversal(
        G=G,
        scores=scores,
        user_input=user_input
    )

    traversal.run()

if __name__ == "__main__":
    main()