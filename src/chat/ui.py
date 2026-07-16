"""Streamlit chat UI components and input handling."""

import streamlit as st

from src.chat.state import clear_chat_history
from src.retrieval.qa import ask_paper_question


def show_chat_with_paper() -> None:
    """Render the Streamlit chat interface for asking questions about the PDF."""
    st.markdown("---")

    chat_header_col, clear_col = st.columns([4, 1])
    with chat_header_col:
        st.subheader("💬 Chat With Paper")
        if st.session_state.paper_name:
            st.caption(f"Active paper: **{st.session_state.paper_name}**")
        st.caption(
            "Ask questions about the uploaded paper. "
            "Answers are grounded in retrieved PDF chunks."
        )
    with clear_col:
        st.write("")
        if st.button(
            "🗑️ Clear Chat",
            key="clear_chat_button",
            use_container_width=True,
            help="Remove chat messages but keep your indexed PDF ready for new questions.",
        ):
            clear_chat_history()
            st.toast("Chat history cleared. Your PDF index is still active.", icon="✅")
            st.rerun()

    # Show previous chat messages.
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # For assistant answers, show which chunks were used.
            if message["role"] == "assistant" and message.get("source_chunks"):
                with st.expander("📎 Source Chunks Used"):
                    for chunk in message["source_chunks"]:
                        chunk_id = chunk.get("chunk_id", "?")
                        st.markdown(f"**Chunk {chunk_id}**")
                        st.write(chunk.get("text", ""))
                        st.markdown("---")


def handle_chat_input() -> None:
    """Render suggested questions and process new chat input."""
    st.markdown("### 💡 Suggested Questions")

    if "selected_question" not in st.session_state:
        st.session_state.selected_question = None

    if st.button("📌 Main Contribution"):
        st.session_state.selected_question = (
            "What is the main contribution of this paper?"
        )

    if st.button("📌 Methodology Used"):
        st.session_state.selected_question = (
            "What methodology was used in this paper?"
        )

    if st.button("📌 Limitations"):
        st.session_state.selected_question = (
            "What are the limitations of this paper?"
        )

    if st.button("📌 Future Work"):
        st.session_state.selected_question = (
            "What future work is suggested by this paper?"
        )

    typed_question = st.chat_input(
        "Ask a question about this paper..."
    )

    user_question = (
        st.session_state.selected_question
        if st.session_state.selected_question
        else typed_question
    )

    if user_question:
        st.session_state.selected_question = None

    # Chat input stays at the bottom of the section.
    if user_question:
        # Save and display the user's message immediately.
        st.session_state.chat_messages.append(
            {"role": "user", "content": user_question}
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate an answer with RAG.
        with st.chat_message("assistant"):
            with st.spinner("Searching the paper and generating an answer..."):
                try:
                    result = ask_paper_question(
                        st.session_state.vector_store,
                        user_question,
                    )

                    answer = result["answer"]
                    source_chunks = result["source_chunks"]

                    st.markdown(answer)

                    if source_chunks:
                        with st.expander("📎 Source Chunks Used"):
                            for chunk in source_chunks:
                                chunk_id = chunk.get("chunk_id", "?")
                                st.markdown(f"**Chunk {chunk_id}**")
                                st.write(chunk.get("text", ""))
                                st.markdown("---")

                    # Save assistant response for future reruns.
                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "source_chunks": source_chunks,
                        }
                    )

                except Exception as error:
                    error_message = f"Something went wrong: {error}"
                    st.error(error_message)
                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                            "source_chunks": [],
                        }
                    )
