import streamlit as st
import pickle
import time
import json
from core.traversal import KG_Traversal
from core.nlu import DDxGraphNLU
from core.parser import Parser

# Page configuration
st.set_page_config(
    page_title="Differential Diagnosis Assistant", page_icon="🩺", layout="wide"
)

# Custom Styles
st.markdown(
    """
<style>
    /* ============================================================
       Diagnostic Console theme
       Ink-dark canvas + faint chart-paper grid + one teal "pulse"
       accent. Percentages, counts, and step numbers use a
       monospace face so they read like instrument readouts.
       ============================================================ */

    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --ink: #06090f;
        --ink-soft: #0b0f17;
        --panel: rgba(255, 255, 255, 0.035);
        --panel-strong: rgba(255, 255, 255, 0.05);
        --panel-border: rgba(255, 255, 255, 0.07);
        --grid-line: rgba(120, 200, 190, 0.05);

        --pulse: #4FD8C4;
        --pulse-soft: #2BB8A6;
        --pulse-dim: rgba(79, 216, 196, 0.14);

        --vital: #3FD68A;
        --vital-dim: rgba(63, 214, 138, 0.12);

        --alert: #F2727B;
        --alert-dim: rgba(242, 114, 123, 0.12);

        --caution: #E7AD52;
        --caution-dim: rgba(231, 173, 82, 0.12);

        --text: #E8EDF1;
        --text-muted: #8C99A8;

        --r-lg: 18px;
        --r-md: 12px;
        --ease: cubic-bezier(0.22, 1, 0.36, 1);
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            transition-duration: 0.001ms !important;
            animation-duration: 0.001ms !important;
        }
    }

    /* ---------- Global canvas ---------- */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'IBM Plex Sans', sans-serif !important;
        color: var(--text) !important;
        background-color: var(--ink) !important;
        background-image:
            repeating-linear-gradient(0deg, var(--grid-line) 0 1px, transparent 1px 44px),
            repeating-linear-gradient(90deg, var(--grid-line) 0 1px, transparent 1px 44px),
            radial-gradient(120% 90% at 50% -10%, rgba(79, 216, 196, 0.06), transparent 60%) !important;
    }

    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* ---------- Headings ---------- */
    h1 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        color: var(--pulse) !important;
        letter-spacing: -0.02em !important;
        text-shadow: 0 0 22px rgba(79, 216, 196, 0.3);
        padding-bottom: 0.6rem;
        border-bottom: 1px solid var(--panel-border);
        margin-bottom: 0.6rem !important;
    }

    h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        color: var(--text) !important;
        letter-spacing: -0.01em !important;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background-color: var(--ink-soft) !important;
        border-right: 1px solid var(--panel-border) !important;
    }

    [data-testid="stSidebar"] h1 {
        font-size: 1.25rem !important;
        border-bottom: none;
        padding-bottom: 0;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.09em !important;
        color: var(--text-muted) !important;
    }

    [data-testid="stSidebar"] p {
        color: var(--text-muted) !important;
        font-size: 0.86rem;
    }

    /* ---------- Sidebar metrics (Yes / No symptom counts) ---------- */
    [data-testid="stMetricLabel"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.68rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted) !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 600 !important;
        color: var(--pulse) !important;
        font-size: 1.7rem !important;
    }

    /* ---------- Step progress bar (sidebar) ---------- */
    [data-testid="stProgress"] [data-testid="stMarkdownContainer"] p {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: var(--text-muted) !important;
    }

    [data-testid="stProgressBarTrack"] {
        background-color: rgba(255, 255, 255, 0.06) !important;
        border-radius: 99px !important;
    }

    [data-testid="stProgressBarTrack"] > div {
        background: linear-gradient(90deg, var(--pulse-soft), var(--pulse)) !important;
        border-radius: 99px !important;
    }

    /* ---------- Sliders ---------- */
    [data-testid="stSlider"] label p {
        color: var(--text-muted) !important;
        font-size: 0.86rem !important;
    }

    [data-testid="stSlider"] [role="slider"] {
        background-color: var(--pulse) !important;
        box-shadow: 0 0 0 4px var(--pulse-dim) !important;
    }

    [data-testid="stSliderThumbValue"] {
        font-family: 'IBM Plex Mono', monospace !important;
        color: var(--pulse) !important;
    }

    [data-testid="stSliderTickBar"] {
        opacity: 0.45;
    }

    /* ---------- Alert boxes (st.info / st.warning) ---------- */
    [data-testid="stAlert"] {
        background-color: var(--panel) !important;
        border: 1px solid var(--panel-border) !important;
        border-radius: var(--r-md) !important;
        color: var(--text) !important;
    }

    /* ---------- Chat messages ---------- */
    [data-testid="stChatMessage"] {
        border-radius: var(--r-lg) !important;
        padding: 1.15rem 1.4rem !important;
        margin-bottom: 1rem !important;
        background-color: var(--panel) !important;
        border: 1px solid var(--panel-border) !important;
        border-left: 3px solid transparent !important;
        transition: border-color 0.2s var(--ease), transform 0.2s var(--ease);
    }

    [data-testid="stChatMessage"]:hover {
        transform: translateY(-1px);
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        border-left-color: var(--pulse) !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        border-left-color: rgba(255, 255, 255, 0.18) !important;
        background-color: rgba(255, 255, 255, 0.02) !important;
    }

    [data-testid="stChatMessageAvatarUser"] {
        background-color: rgba(255, 255, 255, 0.08) !important;
    }

    [data-testid="stChatMessageAvatarAssistant"] {
        background-color: var(--pulse-dim) !important;
        color: var(--pulse) !important;
    }

    /* ---------- Chat input ---------- */
    [data-testid="stChatInput"] {
        background-color: var(--ink-soft) !important;
        border: 1px solid var(--panel-border) !important;
        border-radius: var(--r-md) !important;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.35) !important;
    }

    [data-testid="stChatInput"]:has([data-testid="stChatInputTextArea"]:focus) {
        border-color: var(--pulse) !important;
        box-shadow: 0 0 0 3px var(--pulse-dim) !important;
    }

    [data-testid="stChatInputTextArea"] {
        background-color: transparent !important;
        color: var(--text) !important;
        font-size: 0.95rem !important;
    }

    [data-testid="stChatInputSubmitButton"] {
        background-color: var(--pulse) !important;
        border-radius: 10px !important;
    }

    /* Generic text inputs (initial symptom entry, etc.) */
    textarea, input, [role="textbox"] {
        background-color: var(--ink-soft) !important;
        color: var(--text) !important;
        border-color: var(--panel-border) !important;
    }

    textarea:focus, input:focus, [role="textbox"]:focus {
        border-color: var(--pulse) !important;
        box-shadow: 0 0 0 3px var(--pulse-dim) !important;
        outline: none !important;
    }

    /* ---------- Buttons ---------- */
    [data-testid="stBaseButton-secondary"] {
        border-radius: var(--r-md) !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        background-color: var(--panel-strong) !important;
        border: 1px solid var(--panel-border) !important;
        color: var(--text) !important;
        transition: all 0.2s var(--ease) !important;
    }

    [data-testid="stBaseButton-secondary"]:hover {
        border-color: var(--pulse) !important;
        color: var(--pulse) !important;
        transform: translateY(-1px);
    }

    [data-testid="stBaseButton-primary"] {
        border-radius: var(--r-md) !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        background: linear-gradient(135deg, var(--pulse), var(--pulse-soft)) !important;
        border: none !important;
        color: #04140f !important;
        box-shadow: 0 6px 20px rgba(79, 216, 196, 0.25) !important;
        transition: all 0.2s var(--ease) !important;
    }

    [data-testid="stBaseButton-primary"]:hover {
        box-shadow: 0 10px 28px rgba(79, 216, 196, 0.4) !important;
        transform: translateY(-1px);
    }

    [data-testid="stBaseButton-primary"]:focus-visible,
    [data-testid="stBaseButton-secondary"]:focus-visible {
        outline: 2px solid var(--pulse);
        outline-offset: 2px;
    }

    /* ---------- Condition probability meters (results) ---------- */
    .custom-progress-container {
        background-color: var(--panel) !important;
        border: 1px solid var(--panel-border) !important;
        border-radius: var(--r-md);
        padding: 1rem 1.25rem;
        margin-bottom: 0.85rem;
        transition: border-color 0.25s var(--ease), transform 0.25s var(--ease);
    }

    .custom-progress-container:hover {
        border-color: rgba(79, 216, 196, 0.35);
        transform: translateX(2px);
    }

    .custom-progress-label {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 0.65rem;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.92rem;
        font-weight: 600;
        color: var(--text);
    }

    .custom-progress-label span:last-child {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        color: var(--pulse);
        font-variant-numeric: tabular-nums;
    }

    /* Faint tick marks every 10% — a measurement-scale motif that
       echoes the diagnostic theme rather than a plain flat bar. */
    .custom-progress-bar {
        position: relative;
        height: 7px;
        width: 100%;
        border-radius: 99px;
        overflow: hidden;
        background-color: rgba(255, 255, 255, 0.06);
        background-image: repeating-linear-gradient(
            90deg,
            rgba(255, 255, 255, 0.14) 0,
            rgba(255, 255, 255, 0.14) 1px,
            transparent 1px,
            transparent 10%
        );
    }

    .custom-progress-fill {
        height: 100%;
        border-radius: 99px;
        background: linear-gradient(90deg, var(--pulse-soft), var(--pulse));
        box-shadow: 0 0 10px rgba(79, 216, 196, 0.45);
    }

    /* ---------- Symptom badges ---------- */
    .symptom-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px 4px 9px;
        border-radius: 6px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        margin: 0 6px 6px 0;
    }

    .symptom-badge-yes {
        background-color: var(--vital-dim) !important;
        border: 1px solid rgba(63, 214, 138, 0.3) !important;
        border-left: 3px solid var(--vital) !important;
        color: var(--vital) !important;
    }

    .symptom-badge-no {
        background-color: var(--alert-dim) !important;
        border: 1px solid rgba(242, 114, 123, 0.3) !important;
        border-left: 3px solid var(--alert) !important;
        color: var(--alert) !important;
    }

    /* ---------- Evidence panels (results columns) ---------- */
    /* Streamlit renders the opening/closing markdown calls for this
       class as separate, disconnected elements, so it can't actually
       wrap the content below it like a real card. Styled as a
       deliberate accent rule instead of an empty box that looks broken. */
    .evidence-column-card {
        display: block;
        height: 4px;
        width: 100%;
        border-radius: 99px;
        background: linear-gradient(90deg, var(--pulse), transparent);
        margin-bottom: 0.75rem;
    }

    [data-testid="stMarkdownContainer"] ul {
        margin: 0.1rem 0 0.6rem 0;
        padding-left: 1.1rem;
    }

    [data-testid="stMarkdownContainer"] li {
        color: var(--text-muted);
        margin-bottom: 0.15rem;
    }

    [data-testid="stMarkdownContainer"] li::marker {
        color: var(--pulse);
    }

    [data-testid="stMarkdownContainer"] strong {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text);
    }

    /* ---------- Scrollbars ---------- */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    ::-webkit-scrollbar-track {
        background: var(--ink);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(79, 216, 196, 0.25);
        border-radius: 99px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(79, 216, 196, 0.4);
    }

    /* ---------- Mobile ---------- */
    @media (max-width: 640px) {
        h1 {
            font-size: 1.6rem !important;
        }
        .custom-progress-container,
        [data-testid="stChatMessage"] {
            padding: 0.9rem 1rem !important;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

# Sidebar Settings
st.sidebar.title("🩺 Control Center")
st.sidebar.markdown("Configure traversal options below:")
max_steps = st.sidebar.slider("Maximum follow-up questions", min_value=2, max_value=15, value=5)
top_k_conditions = st.sidebar.slider("Top Conditions to Track", min_value=3, max_value=20, value=10)

st.session_state.max_steps = max_steps
st.session_state.top_k_conditions = top_k_conditions

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

# Safe Loading
try:
    with st.spinner("Loading Knowledge Graph..."):
        G = load_graph()
    nlu, parser = load_nlu_and_parser(G)
    release_evidences = load_release_evidences()
except FileNotFoundError:
    st.error("⚠️ **Knowledge Graph or Evidence files not found.**")
    st.info("Please build the Knowledge Graph by running the `KG_Construction` notebook and download the DDXPlus datasets first.")
    st.stop()

def get_readable_evidence(ev_id: str) -> str:
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

# Sidebar Live Session Metrics
if "traversal" in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Live Diagnostic Session")
    t = st.session_state.traversal
    
    # Progress bar for steps
    current_step = st.session_state.step
    max_steps_val = st.session_state.max_steps
    progress_pct = min(1.0, current_step / max_steps_val)
    st.sidebar.progress(progress_pct, text=f"Step {current_step} / {max_steps_val}")
    
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Yes Symptoms", len(t.observed_yes))
    col2.metric("No Symptoms", len(t.observed_no))
    
    if t.observed_yes:
        st.sidebar.markdown("**Confirmed Symptoms:**")
        badges = []
        for ev in list(t.observed_yes):
            desc = get_readable_evidence(ev)
            badges.append(f'<span class="symptom-badge symptom-badge-yes">✓ {desc}</span>')
        st.sidebar.markdown(" ".join(badges), unsafe_allow_html=True)
    if t.observed_no:
        st.sidebar.markdown("**Absent Symptoms:**")
        badges = []
        for ev in list(t.observed_no):
            desc = get_readable_evidence(ev)
            badges.append(f'<span class="symptom-badge symptom-badge-no">✗ {desc}</span>')
        st.sidebar.markdown(" ".join(badges), unsafe_allow_html=True)

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

    # Manually parse initial query and apply sanitized values
    if user_input:
        context = traversal.nlu.retrieve(user_input)
        evidences, values = traversal.parser.parse_query(user_input, context)
        if evidences:
            clean_ev, clean_val = sanitize_evidence_and_values(traversal, evidences, values)
            if clean_ev:
                traversal.apply_initial_evidence(clean_ev, clean_val)

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

# Main Application Render
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
        # Convert raw scores to probabilities using base class method
        t._convert_scores_to_probabilities(t.scores)
        
        top_candidates = sorted(t.scores.items(), key=lambda x: x[1], reverse=True)[
            : st.session_state.top_k_conditions
        ]

        st.header("Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Top Conditions")
            for c, s in top_candidates:
                percentage = s * 100
                st.markdown(
                    f"""
                    <div class="custom-progress-container">
                        <div class="custom-progress-label">
                            <span>🩺 <strong>{c}</strong></span>
                            <span>{percentage:.1f}%</span>
                        </div>
                        <div class="custom-progress-bar">
                            <div class="custom-progress-fill" style="width: {percentage}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col2:
            st.markdown('<div class="evidence-column-card">', unsafe_allow_html=True)
            st.subheader("✅ Supporting Evidence")
            for c, _ in top_candidates:
                supporting = [e for e in t.observed_yes if t.G.has_edge(c, e)]
                if supporting:
                    st.markdown(f"**{c}**")
                    for ev in supporting:
                        desc = get_readable_evidence(ev)
                        st.markdown(f"- {desc}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="evidence-column-card">', unsafe_allow_html=True)
            st.subheader("⚠️ Missing Evidence")
            for c, _ in top_candidates:
                absent = []
                for e in t.observed_no:
                    if t.G.has_edge(c, e):
                        p = t.G.edges[c, e].get("p_e_given_c", t.SMOOTH)
                        if p >= t.ABSENCE_PROB_THRESHOLD:
                            absent.append(e)
                if absent:
                    st.markdown(f"**{c}**")
                    for ev in absent:
                        desc = get_readable_evidence(ev)
                        st.markdown(f"- {desc}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.divider()
        if st.button("Restart Diagnosis"):
            t = st.session_state.traversal
            if hasattr(t, "save_cache"):
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