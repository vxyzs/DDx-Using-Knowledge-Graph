import streamlit as st
import pickle
import time
from core.traversal import KG_Traversal

st.set_page_config(
    page_title="Differential Diagnosis Assistant", page_icon="🩺", layout="wide"
)

# Basic Premium CSS for Streamlit
st.markdown(
    """
<style>
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    .stChatInputContainer {
        border-radius: 12px;
        background-color: #1e293b;
        color: #e2e8f0;
    }
    .stChatMessage {
        border-radius: 8px;
        padding: 10px;
        background-color: #1e293b;
        color: #e2e8f0;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_graph():
    return pickle.load(open("./Pickle/kg.pkl", "rb"))


with st.spinner("Loading Knowledge Graph..."):
    G = load_graph()


def init_traversal(user_input):
    scores = {c: 0.0 for c in G.nodes if G.nodes[c]["type"] == "condition"}
    traversal = KG_Traversal(G, scores, user_input=user_input)
    st.session_state.traversal = traversal
    st.session_state.step = 0
    st.session_state.max_steps = 5
    st.session_state.top_k_conditions = 10
    st.session_state.chat_history = [{"role": "user", "content": user_input}]
    st.session_state.current_question = None
    st.session_state.current_evidence = None
    st.session_state.is_parent = False
    st.session_state.finished = False


def get_next_question():
    t = st.session_state.traversal
    ranked = sorted(t.scores.items(), key=lambda x: x[1], reverse=True)[
        : st.session_state.top_k_conditions
    ]
    candidate_conditions = [c for c, _ in ranked]

    if (
        len(candidate_conditions) <= 1
        or st.session_state.step >= st.session_state.max_steps
    ):
        st.session_state.finished = True
        return

    evidence = t.get_discriminating_evidence(candidate_conditions)
    if evidence is None:
        st.session_state.finished = True
        return

    parent = t.G.nodes[evidence].get("parent", None)

    if parent and parent != evidence and parent not in t.observed_yes:
        if parent not in t.asked:
            question = t.G.nodes[parent]["question_en"]
            st.session_state.current_question = question
            st.session_state.current_evidence = parent
            st.session_state.is_parent = True
            return
        if parent in t.observed_no:
            t.asked.add(evidence)
            return get_next_question()

    t.asked.add(evidence)
    question = t.G.nodes[evidence]["question_en"]
    st.session_state.current_question = question
    st.session_state.current_evidence = evidence
    st.session_state.is_parent = False


def process_answer(ans):
    t = st.session_state.traversal
    question = st.session_state.current_question
    evidence_id = st.session_state.current_evidence

    context_text = f"The doctor asked: '{question}'. The patient answered: '{ans}'"
    context = t.retrieve(context_text)
    ext_evidences, ext_values = t.parse_query(context_text, context)

    if ext_evidences:
        t.apply_initial_evidence(ext_evidences, ext_values)

    if st.session_state.is_parent:
        t.asked.add(evidence_id)
        # We don't increment step for parent questions
    else:
        st.session_state.step += 1

    get_next_question()


st.title("🩺 AI Diagnostic Traversal")
st.markdown(
    "Interact with the knowledge graph diagnosis assistant to refine your symptoms."
)

if "traversal" not in st.session_state:
    st.info(
        "Welcome. Please describe your symptoms in detail to begin the diagnostic process."
    )

    user_input = st.text_area("Initial Case / Symptoms:", height=100)

    if st.button("Start Diagnosis", type="primary"):
        if user_input.strip():
            with st.spinner("Analyzing initial symptoms..."):
                init_traversal(user_input.strip())
                get_next_question()
            st.rerun()
        else:
            st.warning("Please enter your symptoms to continue.")
else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if st.session_state.finished:
        st.success("Diagnostic Traversal Complete.")

        t = st.session_state.traversal
        top_candidates = sorted(t.scores.items(), key=lambda x: x[1], reverse=True)[
            : st.session_state.top_k_conditions
        ]

        st.header("Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Top Conditions")
            for c, s in top_candidates:
                st.write(f"**{c}**: {s:.4f}")

        with col2:
            st.subheader("Supporting Evidence")
            for c, _ in top_candidates:
                supporting = [e for e in t.observed_yes if t.G.has_edge(c, e)]
                if supporting:
                    st.write(f"*{c}:*")
                    for ev in supporting:
                        st.write(f"- {ev}")

        with col3:
            st.subheader("Missing Evidence")
            for c, _ in top_candidates:
                absent = []
                for e in t.observed_no:
                    if t.G.has_edge(c, e):
                        p = t.G.edges[c, e].get("p_e_given_c", t.SMOOTH)
                        if p >= t.ABSENCE_PROB_THRESHOLD:
                            absent.append(e)
                if absent:
                    st.write(f"*{c}:*")
                    for ev in absent:
                        st.write(f"- {ev}")

        st.divider()
        if st.button("Restart Diagnosis"):
            t = st.session_state.traversal
            t.save_cache()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    else:
        if st.session_state.current_question:
            with st.chat_message("assistant"):
                st.write(st.session_state.current_question)

            ans = st.chat_input("Your answer...")
            if ans:
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": st.session_state.current_question}
                )
                st.session_state.chat_history.append({"role": "user", "content": ans})
                with st.spinner("Understanding response..."):
                    process_answer(ans)
                st.rerun()
