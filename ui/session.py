import pickle
import json
import streamlit as st
from core.traversal import KG_Traversal
from core.nlu import DDxGraphNLU
from core.parser import Parser
from ui.ai import refine_question


# Cached Resources
@st.cache_resource
def load_graph():
    return pickle.load(open("./Pickle/kg.pkl", "rb"))


@st.cache_resource
def load_nlu_and_parser(_G):
    nlu = DDxGraphNLU(_G)
    parser = Parser()
    return nlu, parser


@st.cache_resource
def load_release_evidences():
    with open("Data/ddxplus/release_evidences.compact.json", "r") as f:
        return json.load(f)


def initialize_app_state():
    """
    Ensures that the knowledge graph resources are loaded into state and 
    controls in the sidebar are configured.
    """
    # Safe Loading of KG Resources
    try:
        if "G" not in st.session_state:
            with st.spinner("Loading Knowledge Graph..."):
                st.session_state.G = load_graph()
        if "nlu" not in st.session_state or "parser" not in st.session_state:
            nlu, parser = load_nlu_and_parser(st.session_state.G)
            st.session_state.nlu = nlu
            st.session_state.parser = parser
        if "release_evidences" not in st.session_state:
            st.session_state.release_evidences = load_release_evidences()
    except FileNotFoundError:
        st.error("⚠️ **Knowledge Graph or Evidence files not found.**")
        st.info("Please build the Knowledge Graph by running the `KG_Construction` notebook and download the DDXPlus datasets first.")
        st.stop()

    # Configure controls in sidebar
    st.sidebar.title("🩺 Control Center")
    st.sidebar.markdown("Configure traversal options below:")
    max_steps = st.sidebar.slider("Maximum follow-up questions", min_value=2, max_value=15, value=5)
    top_k_conditions = st.sidebar.slider("Top Conditions to Track", min_value=3, max_value=20, value=10)

    st.session_state.max_steps = max_steps
    st.session_state.top_k_conditions = top_k_conditions


def get_readable_evidence(ev_id: str, release_evidences) -> str:
    """
    Convert an evidence ID (e.g., 'E_91' or 'E_55_@_V_125') to a human-readable description.
    """
    if not ev_id:
        return ev_id
    if not isinstance(ev_id, str):
        return str(ev_id)

    # If it is a categorical/numerical sub-value (e.g., E_55_@_V_125)
    if "_@_" in ev_id:
        parts = ev_id.split("_@_")
        parent_id = parts[0]
        val_id = parts[1]
        
        parent_meta = release_evidences.get(parent_id, {})
        parent_name = parent_meta.get("question_en", parent_id)
        
        # Look up value meaning
        value_meaning = parent_meta.get("value_meaning", {})
        val_meta = value_meaning.get(val_id, {})
        val_name = val_meta.get("en", val_id)
        
        return f"{parent_name} -> **{val_name}**"
    
    # If it's a binary evidence
    meta = release_evidences.get(ev_id, {})
    return meta.get("question_en", ev_id)


def sanitize_evidence_and_values(t, evidences, values):
    """
    Sanitize and validate extracted evidence and values to prevent KeyErrors
    when updating the graph traversal.
    """
    clean_evidences = []
    clean_values = []
    
    for evid, val in zip(evidences, values):
        # 1. Ensure the evidence node exists in the graph
        if evid not in t.G.nodes:
            continue
            
        # 2. Check the data type and format of the value
        if val in ("YES", "NO"):
            clean_evidences.append(evid)
            clean_values.append(val)
        else:
            # If val is a string (e.g. "E_130_@_V_157"), wrap it in a list
            if isinstance(val, str):
                val = [val]
                
            # If val is an iterable, keep only valid sub-values (edges exist in graph)
            if isinstance(val, (list, set, tuple)):
                valid_subvals = [v for v in val if t.G.has_edge(evid, v)]
                if valid_subvals:
                    clean_evidences.append(evid)
                    clean_values.append(valid_subvals)
                    
    return clean_evidences, clean_values


def init_traversal(user_input):
    """
    Sets up a fresh diagnostic traversal.
    """
    G = st.session_state.G
    nlu = st.session_state.nlu
    parser = st.session_state.parser
    
    scores = {c: 0.0 for c in G.nodes if G.nodes[c]["type"] == "condition"}
    
    # Initialize traversal without user_input to prevent instantiation crashes
    traversal = KG_Traversal(G, scores, nlu, parser, user_input=None)
    st.session_state.traversal = traversal
    st.session_state.step = 0
    st.session_state.chat_history = [{"role": "user", "content": user_input}]
    st.session_state.current_question = None
    st.session_state.current_evidence = None
    st.session_state.is_parent = False
    st.session_state.finished = False
    st.session_state.clarification_history = []

    # Manually parse initial query and apply sanitized values
    if user_input:
        context = traversal.nlu.retrieve(user_input)
        evidences, values = traversal.parser.parse_query(user_input, context)
        if evidences:
            clean_ev, clean_val = sanitize_evidence_and_values(traversal, evidences, values)
            if clean_ev:
                traversal.apply_initial_evidence(clean_ev, clean_val)


def get_next_question():
    """
    Finds and prepares the next question using graph entropy calculations.
    """
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
            raw_question = t.G.nodes[parent]["question_en"]
            st.session_state.current_question = refine_question(raw_question)
            st.session_state.current_evidence = parent
            st.session_state.is_parent = True
            st.session_state.clarification_history = []
            return
        if parent in t.observed_no:
            t.asked.add(evidence)
            return get_next_question()

    t.asked.add(evidence)
    raw_question = t.G.nodes[evidence]["question_en"]
    st.session_state.current_question = refine_question(raw_question)
    st.session_state.current_evidence = evidence
    st.session_state.is_parent = False
    st.session_state.clarification_history = []


def process_user_answer(ans):
    """
    Injects the parsed user answer back into the traversal scoring matrix.
    """
    t = st.session_state.traversal
    question = st.session_state.current_question
    evidence_id = st.session_state.current_evidence

    # Fall back to NLU and LLM parsing
    context_text = f"The doctor asked: '{question}'. The patient answered: '{ans}'"
    context = t.nlu.retrieve(context_text)
    ext_evidences, ext_values = t.parser.parse_query(context_text, context)

    if ext_evidences:
        clean_ev, clean_val = sanitize_evidence_and_values(t, ext_evidences, ext_values)
        if clean_ev:
            t.apply_initial_evidence(clean_ev, clean_val)

    if st.session_state.is_parent:
        t.asked.add(evidence_id)
    
    # Increment step for every follow-up question asked
    st.session_state.step += 1

    get_next_question()
