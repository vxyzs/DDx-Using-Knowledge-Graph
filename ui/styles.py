CSS_STYLE = """
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

    /* ---------- Clinical Card Details Accordion ---------- */
    details.clinical-card {
        background-color: var(--panel) !important;
        border: 1px solid var(--panel-border) !important;
        border-radius: var(--r-md);
        margin-bottom: 1rem;
        padding: 1.15rem 1.4rem;
        transition: border-color 0.25s var(--ease), box-shadow 0.25s var(--ease);
        cursor: pointer;
    }
    details.clinical-card:hover {
        border-color: rgba(79, 216, 196, 0.25);
    }
    details.clinical-card[open] {
        border-color: rgba(79, 216, 196, 0.45);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        background-color: rgba(255, 255, 255, 0.045) !important;
    }
    summary.clinical-summary {
        display: flex;
        justify-content: space-between;
        align-items: center;
        outline: none;
        user-select: none;
    }
    summary.clinical-summary::-webkit-details-marker {
        display: none;
    }
    .summary-header-left {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .condition-icon {
        font-size: 1.25rem;
    }
    .condition-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text);
    }
    .summary-header-right {
        display: flex;
        align-items: center;
        gap: 1.25rem;
    }
    .probability-badge {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        color: var(--pulse);
        font-size: 0.95rem;
        background-color: var(--pulse-dim);
        padding: 2px 8px;
        border-radius: 6px;
        border: 1px solid rgba(79, 216, 196, 0.2);
    }
    .progress-bar-mini {
        position: relative;
        height: 8px;
        width: 120px;
        border-radius: 99px;
        overflow: hidden;
        background-color: rgba(255, 255, 255, 0.06);
    }
    .progress-fill-mini {
        height: 100%;
        border-radius: 99px;
        background: linear-gradient(90deg, var(--pulse-soft), var(--pulse));
        box-shadow: 0 0 8px rgba(79, 216, 196, 0.4);
    }
    .expand-icon {
        color: var(--text-muted);
        font-size: 0.75rem;
        transition: transform 0.25s var(--ease);
    }
    details.clinical-card[open] .expand-icon {
        transform: rotate(180deg);
    }
    .card-details-content {
        margin-top: 1.25rem;
        padding-top: 1.25rem;
        border-top: 1px solid var(--panel-border);
        cursor: default;
    }
    /* AI Explanation Grid */
    .ai-explanation-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-bottom: 1.25rem;
    }
    .explanation-box {
        background-color: rgba(255, 255, 255, 0.015);
        border: 1px solid var(--panel-border);
        border-radius: var(--r-md);
        padding: 1rem;
    }
    .explanation-box h5 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 0;
        margin-bottom: 0.5rem;
        color: var(--text);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .explanation-box p {
        font-size: 0.88rem;
        color: var(--text-muted);
        margin: 0;
        line-height: 1.45;
    }
    .suspect-box {
        border-left: 3px solid var(--vital) !important;
    }
    .missing-box {
        border-left: 3px solid var(--caution) !important;
    }
    .steps-box {
        grid-column: span 2;
        border-left: 3px solid var(--pulse) !important;
        background-color: rgba(79, 216, 196, 0.02) !important;
    }
    /* Evidence lists in KG */
    .evidence-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
    }
    .evidence-col {
        background-color: rgba(255, 255, 255, 0.01);
        border: 1px solid var(--panel-border);
        border-radius: var(--r-md);
        padding: 1rem;
    }
    .evidence-col h5 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.88rem;
        font-weight: 600;
        margin-top: 0;
        margin-bottom: 0.75rem;
        color: var(--text);
    }
    .evidence-col ul {
        margin: 0;
        padding-left: 1.1rem;
        list-style-type: none;
    }
    .evidence-col li {
        font-size: 0.84rem;
        color: var(--text-muted);
        margin-bottom: 0.4rem;
        position: relative;
    }
    .evidence-col li::before {
        content: "•";
        color: var(--pulse);
        font-weight: bold;
        display: inline-block;
        width: 1em;
        margin-left: -1em;
    }
    @media (max-width: 768px) {
        .ai-explanation-grid,
        .evidence-grid {
            grid-template-columns: 1fr;
        }
        .steps-box {
            grid-column: span 1;
        }
        .summary-header-right {
            gap: 0.5rem;
        }
        .progress-bar-mini {
            display: none;
        }
    }

    /* ---------- Homepage Styles ---------- */
    .hero-container {
        text-align: center;
        padding: 2.75rem 1.5rem;
        background: radial-gradient(circle at 50% 30%, rgba(79, 216, 196, 0.08) 0%, transparent 70%);
        border-radius: var(--r-lg);
        margin-bottom: 2.25rem;
        border: 1px solid rgba(79, 216, 196, 0.07);
    }
    .hero-badge {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--pulse);
        background-color: var(--pulse-dim);
        padding: 5px 15px;
        border-radius: 99px;
        border: 1px solid rgba(79, 216, 196, 0.22);
        margin-bottom: 1.15rem;
        box-shadow: 0 0 18px rgba(79, 216, 196, 0.12);
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.85rem !important;
        font-weight: 700 !important;
        color: var(--text);
        margin-bottom: 0.65rem !important;
        letter-spacing: -0.03em !important;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: var(--text-muted);
        max-width: 700px;
        margin: 0 auto 0 auto;
        line-height: 1.55;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.25rem;
        margin-bottom: 2.25rem;
    }
    .feature-card {
        background-color: var(--panel) !important;
        border: 1px solid var(--panel-border) !important;
        border-radius: var(--r-md);
        padding: 1.5rem;
        transition: border-color 0.25s var(--ease), transform 0.25s var(--ease);
    }
    .feature-card:hover {
        border-color: rgba(79, 216, 196, 0.28);
        transform: translateY(-2px);
    }
    .feature-card-icon {
        font-size: 1.85rem;
        margin-bottom: 0.65rem;
    }
    .feature-card h4 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 0;
        margin-bottom: 0.45rem;
        color: var(--text);
    }
    .feature-card p {
        font-size: 0.86rem;
        color: var(--text-muted);
        margin: 0;
        line-height: 1.48;
    }
    .console-container {
        background-color: var(--ink-soft) !important;
        border: 1px solid var(--panel-border) !important;
        border-radius: var(--r-lg);
        padding: 2rem;
        box-shadow: 0 18px 45px rgba(0,0,0,0.45);
        margin-bottom: 2rem;
    }
    .console-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1.35rem;
        border-bottom: 1px solid var(--panel-border);
        padding-bottom: 0.8rem;
    }
    .console-dot {
        width: 10px;
        height: 10px;
        border-radius: 99px;
    }
    .console-dot-red { background-color: #F2727B; }
    .console-dot-yellow { background-color: #E7AD52; }
    .console-dot-green { background-color: #3FD68A; }
    .console-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-left: 0.5rem;
    }
    .sample-cases-container {
        margin-top: 1.35rem;
        border-top: 1px dashed var(--panel-border);
        padding-top: 1.15rem;
    }
    .sample-case-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 0.75rem;
    }

    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.15rem !important;
        }
        .feature-grid {
            grid-template-columns: 1fr;
        }
    }

    /* ---------- Performance Section ---------- */
    .perf-container {
        background-color: var(--panel) !important;
        border: 1px solid var(--panel-border) !important;
        border-radius: var(--r-lg);
        padding: 2rem;
        margin-bottom: 2.25rem;
    }
    .perf-header {
        margin-bottom: 1.25rem;
    }
    .perf-badge {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--vital);
        background-color: var(--vital-dim);
        padding: 4px 10px;
        border-radius: 4px;
        border: 1px solid rgba(63, 214, 138, 0.25);
        margin-bottom: 0.75rem;
    }
    .perf-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--text);
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    .perf-subtitle {
        font-size: 0.92rem;
        color: var(--text-muted);
        line-height: 1.5;
        margin: 0 0 1.25rem 0;
    }
    .perf-callout {
        background-color: rgba(79, 216, 196, 0.02) !important;
        border: 1px solid rgba(79, 216, 196, 0.12) !important;
        border-left: 3px solid var(--pulse) !important;
        border-radius: var(--r-md);
        padding: 1rem;
        margin-bottom: 1.5rem;
    }
    .perf-callout-text {
        font-size: 0.86rem;
        color: var(--text-muted);
        margin: 0;
        line-height: 1.48;
    }
    .perf-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 1rem;
        margin-bottom: 1.25rem;
    }
    .perf-card {
        background-color: rgba(255, 255, 255, 0.015) !important;
        border: 1px solid var(--panel-border) !important;
        border-radius: var(--r-md);
        padding: 1.15rem 1rem;
        text-align: center;
        transition: border-color 0.25s var(--ease), transform 0.25s var(--ease), box-shadow 0.25s var(--ease);
    }
    .perf-card:hover {
        border-color: rgba(79, 216, 196, 0.35);
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(79, 216, 196, 0.08);
    }
    .perf-card-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .perf-card-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.55rem;
        font-weight: 600;
        color: var(--pulse);
        margin-bottom: 0.25rem;
    }
    .perf-card-label {
        font-size: 0.78rem;
        color: var(--text-muted);
        font-weight: 500;
        line-height: 1.35;
    }
    .perf-interpretation {
        font-size: 0.88rem;
        color: var(--text-muted);
        font-style: italic;
        line-height: 1.48;
        margin: 0;
    }
    @media (max-width: 1024px) {
        .perf-grid {
            grid-template-columns: repeat(3, 1fr);
        }
    }
    @media (max-width: 640px) {
        .perf-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
</style>
"""

