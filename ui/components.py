import streamlit as st
from markdown import markdown
from ui.session import get_readable_evidence, init_traversal, get_next_question, process_user_answer
from ui.ai import generate_explanation, answer_clarification_question
from ui.utils import parse_markdown_explanation, find_matching_row


def render_sidebar():
    """
    Renders live session diagnostic metrics and confirmed/absent symptom badges in the sidebar.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Live Diagnostic Session")
    t = st.session_state.traversal
    release_evidences = st.session_state.release_evidences
    
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
            desc = get_readable_evidence(ev, release_evidences)
            badges.append(f'<span class="symptom-badge symptom-badge-yes">✓ {desc}</span>')
        st.sidebar.markdown(" ".join(badges), unsafe_allow_html=True)
    if t.observed_no:
        st.sidebar.markdown("**Absent Symptoms:**")
        badges = []
        for ev in list(t.observed_no):
            desc = get_readable_evidence(ev, release_evidences)
            badges.append(f'<span class="symptom-badge symptom-badge-no">✗ {desc}</span>')
        st.sidebar.markdown(" ".join(badges), unsafe_allow_html=True)


def render_welcome_screen():
    """
    Renders an attractive clinical homepage featuring a Hero banner,
    interactive features grid, sample symptoms pre-fills, and diagnostic controls.
    """
    # 1. Hero Section
    st.markdown(
        """
        <div class="hero-container">
            <span class="hero-badge">Clinical Decision Support System</span>
            <h1 class="hero-title">AI Diagnostic Traversal Console</h1>
            <p class="hero-subtitle">
                An advanced clinical assistant leveraging disease-evidence medical knowledge graphs 
                and semantic intent parsing to interactively explore patient cases.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. Features/Capabilities Grid
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-card-icon">🧠</div>
                <h4>Knowledge Graph Traversal</h4>
                <p>Calculates dynamic disease probabilities using entropy reduction to guide follow-up questions.</p>
            </div>
            <div class="feature-card">
                <div class="feature-card-icon">💬</div>
                <h4>Natural Language Parsing</h4>
                <p>Converts free-text case descriptions into clinical evidence codes using contextual semantic retrieval.</p>
            </div>
            <div class="feature-card">
                <div class="feature-card-icon">🤖</div>
                <h4>Explainable AI Summary</h4>
                <p>Generates detailed clinical summaries mapping symptoms directly back to disease pathophysiologies.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2.5 Performance & Validation Section
    st.markdown(
        """
        <div class="perf-container">
            <div class="perf-header">
                <span class="perf-badge">Evaluated under Partial Information Conditions</span>
                <h3 class="perf-title">Validated Diagnostic Performance</h3>
                <p class="perf-subtitle">
                    Our system is evaluated under a partial-information diagnostic protocol that simulates real clinical encounters, 
                    where not all patient evidence is available at the start of inference.
                </p>
            </div>
            <div class="perf-callout">
                <p class="perf-callout-text">
                    <strong>Partial-information protocol:</strong> Each clinical case is initialized with 60% of the total available evidence. 
                    The remaining 40% is treated as unobserved and must be discovered through iterative diagnostic reasoning, 
                    mimicking real-world clinical assessment.
                </p>
            </div>
            <div class="perf-grid">
                <div class="perf-card">
                    <div class="perf-card-icon">🎯</div>
                    <div class="perf-card-value">72.43%</div>
                    <div class="perf-card-label">Top-1 Accuracy</div>
                </div>
                <div class="perf-card">
                    <div class="perf-card-icon">📊</div>
                    <div class="perf-card-value">83.46%</div>
                    <div class="perf-card-label">Top-3 Accuracy</div>
                </div>
                <div class="perf-card">
                    <div class="perf-card-icon">📈</div>
                    <div class="perf-card-value">87.04%</div>
                    <div class="perf-card-label">Top-5 Accuracy</div>
                </div>
                <div class="perf-card">
                    <div class="perf-card-icon">🧮</div>
                    <div class="perf-card-value">0.7937</div>
                    <div class="perf-card-label">MRR</div>
                </div>
                <div class="perf-card">
                    <div class="perf-card-icon">🏷️</div>
                    <div class="perf-card-value">2.921</div>
                    <div class="perf-card-label">Mean Rank</div>
                </div>
            </div>
            <p class="perf-interpretation">
                Results demonstrate strong diagnostic ranking performance under incomplete-information settings, 
                with the correct diagnosis appearing within the top three predictions in over 83% of cases.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 3. Interactive Case Entry & Pre-fills
    # st.markdown('<div class="console-container">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="console-header">
            <span class="console-title">Diagnostic Core Initializer v1.0</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Symptom input box
    user_input = st.text_area(
        "Describe Case History / Symptoms in detail:",
        height=120,
        placeholder="e.g. Patient presents with sudden chest tightness, radiating pain to left shoulder, and shortness of breath..."
    )

    # Action button
    btn_click = st.button("Start Diagnosis", type="primary")

    if btn_click:
        if user_input.strip():
            with st.spinner("Analyzing case profile and loading questions..."):
                init_traversal(user_input.strip())
                get_next_question()
            st.rerun()
        else:
            st.warning("Please describe your symptoms to continue.")

    st.markdown('</div>', unsafe_allow_html=True)


def render_chat_interface():
    """
    Renders the active chat dialogue stream, clarification assistant form, and the main chat input.
    """
    col_title, col_restart = st.columns([5, 1.2])
    with col_title:
        st.subheader("💬 Live Diagnostic Conversation")
    with col_restart:
        if st.button("🔄 Restart Diagnosis", key="chat_restart_btn", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Render chat message history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if st.session_state.current_question:
        with st.chat_message("assistant"):
            st.write(st.session_state.current_question)

        # Collapsible Clarification Assistant Chat Box
        with st.expander("❓ Don't understand the question? Ask here!", expanded=False):
            st.markdown(
                "<p style='font-size:0.86rem; color:var(--text-muted); margin-bottom:0.8rem;'>"
                "If you are unsure what the question means or how to check for this symptom, ask below.</p>",
                unsafe_allow_html=True
            )
            
            if "clarification_history" not in st.session_state:
                st.session_state.clarification_history = []
            
            # Render clarification chat stream
            for msg in st.session_state.clarification_history:
                if msg["role"] == "user":
                    st.markdown(f"**🙋 You:** {msg['content']}")
                else:
                    st.markdown(f"**🤖 Assistant:** {msg['content']}")
            
            # Clarification input form
            with st.form(key="clarify_form", clear_on_submit=True):
                col_input, col_btn = st.columns([5, 1])
                with col_input:
                    clarify_q = st.text_input(
                        "Ask a question...",
                        placeholder="e.g., What does this symptom feel like?",
                        label_visibility="collapsed"
                    )
                with col_btn:
                    submit_clarification = st.form_submit_button("Explain")
                    
                if submit_clarification and clarify_q.strip():
                    st.session_state.clarification_history.append({"role": "user", "content": clarify_q.strip()})
                    with st.spinner("Explaining..."):
                        explanation_ans = answer_clarification_question(
                            st.session_state.current_question,
                            st.session_state.current_evidence,
                            clarify_q.strip(),
                            st.session_state.release_evidences
                        )
                        st.session_state.clarification_history.append({"role": "assistant", "content": explanation_ans})
                    st.rerun()

        ans = st.chat_input("Your answer...")
        if ans:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": st.session_state.current_question}
            )
            st.session_state.chat_history.append({"role": "user", "content": ans})
            with st.spinner("Understanding response..."):
                process_user_answer(ans)
            st.rerun()


def render_clinical_report():
    """
    Renders the final unified differential diagnosis report, clinical cards, and restart controls.
    """
    col_title, col_restart = st.columns([5, 1.2])
    with col_title:
        st.success("Diagnostic Traversal Complete.")
    with col_restart:
        if st.button("🔄 Restart Diagnosis", key="report_restart_btn", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    t = st.session_state.traversal
    release_evidences = st.session_state.release_evidences
    
    # Convert raw scores to probabilities using base class method
    t._convert_scores_to_probabilities(t.scores)
    results = t.get_diagnostic_results(st.session_state.top_k_conditions)

    if "explanation" not in st.session_state:
        with st.spinner("Generating explainable AI summary..."):
            # Convert raw evidence IDs to human-readable symptoms before passing to the AI
            readable_results = {
                "top_conditions_with_probabilities": {
                    c: f"{s * 100:.1f}%"
                    for c, s in results["top_conditions_with_scores"].items()
                },
                "supporting_evidence_by_condition": {
                    c: [get_readable_evidence(ev, release_evidences) for ev in supporting]
                    for c, supporting in results["cond_evidence_map"].items()
                },
                "missing_evidence_by_condition": {
                    c: [get_readable_evidence(ev, release_evidences) for ev in absent]
                    for c, absent in results["missing_evidence_map"].items()
                },
            }
            st.session_state.explanation = generate_explanation(readable_results)

    # Parse the AI explanation
    table_rows = []
    summary = ""
    disclaimer = ""
    try:
        table_rows, summary, disclaimer = parse_markdown_explanation(
            st.session_state.explanation
        )
    except Exception as e:
        # Fallback in case of parsing issues
        summary = st.session_state.explanation

    st.header("📋 Clinical Diagnostic Report")

    # Display clinical summary
    summary_html = markdown(summary)
    if summary_html:
        st.markdown(f"""
        <div style="background-color: rgba(79, 216, 196, 0.03);
                    border: 1px solid rgba(79, 216, 196, 0.25);
                    border-left: 4px solid var(--pulse);
                    padding: 1.25rem;
                    border-radius: var(--r-md);
                    margin-bottom: 1.5rem;">
            <h4 style="margin-top: 0; color: var(--pulse); margin-bottom: 0.5rem;">
                🩺 Clinical Summary
            </h4>
            {summary_html}
        </div>
        """, unsafe_allow_html=True)

    # Display candidate conditions as accordion cards
    for c, s in results["top_conditions_with_scores"].items():
        percentage = s * 100
        
        # Find matching AI explanation row
        ai_row = find_matching_row(c, table_rows)
        why_suspected = ai_row[2] if ai_row and len(ai_row) > 2 else "No specific context available."
        ai_missing = ai_row[3] if ai_row and len(ai_row) > 3 else "No contradicting factors highlighted."
        next_steps = ai_row[4] if ai_row and len(ai_row) > 4 else "Consult a healthcare professional."

        # Supporting and missing evidence strings for Knowledge Graph section
        supporting_ev = results["cond_evidence_map"].get(c, [])
        absent_ev = results["missing_evidence_map"].get(c, [])

        supporting_html = ""
        if supporting_ev:
            supporting_html = "\n".join(
                f"<li><strong>{get_readable_evidence(ev, release_evidences)}</strong> (Answered: <em>{t.evidence_answers.get(ev, 'YES')}</em>)</li>"
                for ev in supporting_ev
            )
        else:
            supporting_html = "<li><em>No confirmed symptoms match this condition.</em></li>"

        absent_html = ""
        if absent_ev:
            absent_html = "\n".join(
                f"<li><strong>{get_readable_evidence(ev, release_evidences)}</strong> (Answered: <em>{t.evidence_answers.get(ev, 'NO')}</em>)</li>"
                for ev in absent_ev
            )
        else:
            absent_html = "<li><em>No missing symptoms recorded.</em></li>"

        card_html = f"""
        <details class="clinical-card">
            <summary class="clinical-summary">
                <div class="summary-header-left">
                    <span class="condition-icon">🩺</span>
                    <span class="condition-name">{c}</span>
                </div>
                <div class="summary-header-right">
                    <span class="probability-badge">{percentage:.1f}%</span>
                    <div class="progress-bar-mini">
                        <div class="progress-fill-mini" style="width: {percentage}%;"></div>
                    </div>
                    <span class="expand-icon">▼</span>
                </div>
            </summary>
            <div class="card-details-content">
                <div class="ai-explanation-grid">
                    <div class="explanation-box suspect-box">
                        <h5>🔍 Clinical Reasoning (AI)</h5>
                        <p>{why_suspected}</p>
                    </div>
                    <div class="explanation-box missing-box">
                        <h5>⚠️ Absent / Contradicting Indicators</h5>
                        <p>{ai_missing}</p>
                    </div>
                    <div class="explanation-box steps-box">
                        <h5>📋 Recommended Next Steps</h5>
                        <p>{next_steps}</p>
                    </div>
                </div>
                <div class="evidence-grid">
                    <div class="evidence-col">
                        <h5>✅ Supporting Evidence (Knowledge Graph)</h5>
                        <ul>
                            {supporting_html}
                        </ul>
                    </div>
                    <div class="evidence-col">
                        <h5>❌ Absent / Missing Evidence (Knowledge Graph)</h5>
                        <ul>
                            {absent_html}
                        </ul>
                    </div>
                </div>
            </div>
        </details>
        """
        st.markdown(card_html, unsafe_allow_html=True)

    if disclaimer:
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 2rem; padding: 1rem; border-top: 1px solid var(--panel-border); color: var(--text-muted); font-size: 0.8rem; font-style: italic;">
                {disclaimer}
            </div>
            """,
            unsafe_allow_html=True,
        )


