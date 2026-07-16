"""Streamlit session state helpers for chat and paper data."""

import streamlit as st


def reset_paper_state() -> None:
    """Clear old paper data when a new PDF is uploaded."""
    st.session_state.analysis = None
    st.session_state.vector_store = None
    st.session_state.chat_messages = []
    st.session_state.paper_name = None
    st.session_state.pdf_details = None
    st.session_state.indexing_stats = None


def clear_chat_history() -> None:
    """
    Clear only the chat messages.

    The PDF index (vector_store), analysis, and upload details stay in
    session_state so the user does not need to re-upload or re-analyze.
    """
    st.session_state.chat_messages = []
