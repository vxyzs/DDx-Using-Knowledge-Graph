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

    traversal.run()

if __name__ == "__main__":
    main()