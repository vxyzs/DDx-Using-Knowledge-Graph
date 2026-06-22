import streamlit as st
from ui.styles import CSS_STYLE
from ui.session import initialize_app_state
from ui.components import render_sidebar, render_welcome_screen, render_chat_interface, render_clinical_report

# Page configuration
st.set_page_config(
    page_title="Differential Diagnosis Assistant", page_icon="🩺", layout="wide"
)

# Render Custom Styles
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# Initialize data resources and traversal settings
initialize_app_state()

# Main Application Render Flow
st.title("🩺 AI Diagnostic Traversal")
st.markdown(
    "Interact with the knowledge graph diagnosis assistant to refine your symptoms."
)

if "traversal" not in st.session_state:
    render_welcome_screen()
else:
    # Sidebar Live Metrics
    render_sidebar()

    if st.session_state.finished:
        render_clinical_report()
    else:
        render_chat_interface()