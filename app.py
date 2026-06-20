"""AI Research Assistant - Streamlit app with RAG chat."""

import streamlit as st

from gemini_summarizer import analyze_research_paper
from pdf_extractor import extract_pdf_details
from rag_chat import ask_paper_question
from rag_store import (
    INDEXING_CHAR_LIMIT,
    MAX_ANALYSIS_CHARS,
    build_vector_store_from_chunks,
    create_text_chunks,
    get_text_processing_stats,
    trim_text_for_analysis,
)

APP_MODEL = "Gemini 2.5 Flash"

st.set_page_config(
    page_title="ScholarAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
if st.button("Test Gemini"):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content("Say hello")
        st.success(response.text)
    except Exception as e:
        st.error(str(e))

# ---------------------------------------------------------------------------
# Custom styling for a clean, beginner-friendly layout.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .hero-title {
            font-size: 2.6rem;
            font-weight: 800;
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.35rem;
        }
        .hero-subtitle {
            color: #64748b;
            font-size: 1.15rem;
            line-height: 1.6;
            margin-bottom: 1.75rem;
        }
        .stat-card {
            background: linear-gradient(145deg, #f8fafc 0%, #eef2ff 100%);
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1rem 1.1rem;
            text-align: center;
            min-height: 95px;
        }
        .stat-label {
            color: #64748b;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.35rem;
        }
        .stat-value {
            color: #1e293b;
            font-size: 1.15rem;
            font-weight: 700;
        }
        .difficulty-badge {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.5rem 1.25rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 1rem;
        }
        .sidebar-section {
            margin-bottom: 1.25rem;
        }
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — project info, features, and tech stack.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 ScholarAI")
    st.markdown(
        '<p style="color:#64748b; font-size:0.95rem; margin-top:-0.5rem;">'
        "Your friendly AI research companion"
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 📁 Project Name")
    st.info("**ScholarAI** — AI Research Assistant")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### ✨ Features")
    st.markdown(
        """
        - 📄 **PDF Upload** — Load any research paper
        - 🔍 **Smart Analysis** — Summaries, quizzes, and more
        - 💬 **RAG Chat** — Ask questions about your paper
        - 📎 **Source Chunks** — See where answers come from
        - 🎓 **Beginner Friendly** — ELI15 explanations included
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 🛠️ Tech Stack")
    st.markdown(
        """
        - **Streamlit** — Web interface
        - **Google Gemini** — AI analysis & chat
        - **LangChain** — Text chunking & embeddings
        - **FAISS** — Vector similarity search
        - **PyPDF** — PDF text extraction
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.caption(
        f"💡 Tip: Large PDFs are trimmed to {MAX_ANALYSIS_CHARS:,} characters "
        f"for analysis and {INDEXING_CHAR_LIMIT:,} characters for chat search."
    )

# ---------------------------------------------------------------------------
# Session state keeps data alive across Streamlit reruns.
# Without this, chat history and the vector database would disappear
# every time the user sends a new message.
# ---------------------------------------------------------------------------
if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "paper_name" not in st.session_state:
    st.session_state.paper_name = None

if "pdf_details" not in st.session_state:
    st.session_state.pdf_details = None

if "indexing_stats" not in st.session_state:
    st.session_state.indexing_stats = None


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


def sanitize_paper_filename(paper_name: str) -> str:
    """Turn a PDF filename into a safe name for the analysis export file."""
    base_name = paper_name.rsplit(".", 1)[0] if "." in paper_name else paper_name
    safe_name = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in base_name
    )
    return safe_name.strip("_") or "paper"


def format_analysis_export(analysis: dict, paper_name: str) -> str:
    """Build a clean, readable text export of all analysis sections."""
    lines = [
        "ScholarAI — Research Paper Analysis",
        "=" * 60,
        f"Paper: {paper_name}",
        "",
        "DIFFICULTY RATING",
        "-" * 60,
        f"{analysis.get('difficulty_rating', 'N/A')} / 10",
        "",
        "EXECUTIVE SUMMARY",
        "-" * 60,
        analysis.get("executive_summary", "Not available."),
        "",
        "KEY TAKEAWAYS",
        "-" * 60,
    ]

    takeaways = analysis.get("key_takeaways", [])
    if takeaways:
        for index, takeaway in enumerate(takeaways, start=1):
            lines.append(f"{index}. {takeaway}")
    else:
        lines.append("Not available.")

    lines.extend(
        [
            "",
            "BEGINNER EXPLANATION (ELI15)",
            "-" * 60,
            analysis.get("beginner_explanation", "Not available."),
            "",
            "QUIZ QUESTIONS",
            "-" * 60,
        ]
    )

    quiz = analysis.get("quiz_questions", [])
    if quiz:
        for index, item in enumerate(quiz, start=1):
            question = item.get("question", "No question provided.")
            answer = item.get("answer", "No answer provided.")
            lines.append(f"{index}. {question}")
            lines.append(f"   Answer: {answer}")
            lines.append("")
    else:
        lines.append("Not available.")
        lines.append("")

    lines.extend(["INTERVIEW QUESTIONS", "-" * 60])

    interview = analysis.get("interview_questions", [])
    if interview:
        for index, question in enumerate(interview, start=1):
            lines.append(f"{index}. {question}")
    else:
        lines.append("Not available.")

    lines.extend(["", "FUTURE RESEARCH IDEAS", "-" * 60])

    ideas = analysis.get("future_research_ideas", [])
    if ideas:
        for index, idea in enumerate(ideas, start=1):
            lines.append(f"{index}. {idea}")
    else:
        lines.append("Not available.")

    return "\n".join(lines)


def format_file_size(size_bytes: int) -> str:
    """Turn raw byte count into a readable file size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def show_metrics_row() -> None:
    """Display key app metrics in a clean row at the top of the page."""
    details = st.session_state.pdf_details or {}
    indexing = st.session_state.indexing_stats or {}

    pages = details.get("page_count") or 0
    chunks = indexing.get("chunks_created") or 0
    questions_asked = sum(
        1 for message in st.session_state.chat_messages
        if message["role"] == "user"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="📑 Pages", value=pages)
    with col2:
        st.metric(label="🧩 Chunks Created", value=chunks)
    with col3:
        st.metric(label="💬 Questions Asked", value=questions_asked)
    with col4:
        st.metric(label="🤖 Model Used", value=APP_MODEL)


def show_character_processing_stats(char_count: int) -> None:
    """Show how much of the PDF will be analyzed and indexed."""
    indexing = st.session_state.indexing_stats or {}

    if indexing.get("total_chars") is not None:
        total_chars = indexing["total_chars"]
        chars_analyzed = indexing.get(
            "chars_analyzed",
            min(total_chars, MAX_ANALYSIS_CHARS),
        )
        chars_indexed = indexing.get(
            "chars_indexed",
            min(total_chars, INDEXING_CHAR_LIMIT),
        )
    else:
        stats = get_text_processing_stats(char_count)
        total_chars = stats["total_chars"]
        chars_analyzed = stats["chars_analyzed"]
        chars_indexed = stats["chars_indexed"]

    st.markdown("#### 🔤 Character Processing")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Characters", f"{total_chars:,}")
    with col2:
        st.metric("Characters Analyzed", f"{chars_analyzed:,}")
    with col3:
        st.metric("Characters Indexed", f"{chars_indexed:,}")


def show_large_pdf_warning(char_count: int) -> None:
    """Warn when the PDF exceeds the analysis character limit."""
    if char_count > MAX_ANALYSIS_CHARS:
        st.warning(
            "Large PDF detected. To stay within AI model limits, only the "
            "first portion of the document will be analyzed and indexed."
        )


def run_paper_analysis(uploaded_file, char_count: int) -> None:
    """Run the full analysis pipeline with step-by-step progress indicators."""
    progress_bar = st.progress(0, text="Starting paper analysis...")

    with st.status("Analyzing paper...", expanded=True) as status:
        try:
            st.write("1. Extracting PDF text...")
            progress_bar.progress(10, text="Step 1 of 5: Extracting PDF text")

            if st.session_state.pdf_details is None:
                st.session_state.pdf_details = extract_pdf_details(uploaded_file)

            text = st.session_state.pdf_details["text"]
            if not text.strip():
                status.update(label="Analysis failed", state="error")
                progress_bar.empty()
                st.error("No text could be extracted from this PDF.")
                return

            analysis_text = trim_text_for_analysis(text)

            st.write("2. Creating chunks...")
            progress_bar.progress(30, text="Step 2 of 5: Creating chunks")
            chunk_result = create_text_chunks(text)
            st.caption(f"Created {chunk_result['chunks_created']} chunks.")

            st.write("3. Building vector database...")
            progress_bar.progress(55, text="Step 3 of 5: Building vector database")
            vector_store = build_vector_store_from_chunks(chunk_result["chunks"])

            st.write("4. Generating analysis...")
            progress_bar.progress(75, text="Step 4 of 5: Generating analysis")
            analysis = analyze_research_paper(analysis_text)

            st.write("5. Completed")
            progress_bar.progress(100, text="Step 5 of 5: Completed")
            status.update(
                label="Analysis complete!",
                state="complete",
                expanded=False,
            )

            st.session_state.analysis = analysis
            st.session_state.vector_store = vector_store
            st.session_state.indexing_stats = {
                "chunks_created": chunk_result["chunks_created"],
                "chars_indexed": chunk_result["chars_indexed"],
                "was_truncated": chunk_result["was_truncated"],
                "total_chars": char_count,
                "chars_analyzed": len(analysis_text),
            }
            st.session_state.chat_messages = []

            st.success("✅ Vector database created successfully!")
            #st.rerun()

        except ValueError as error:
            status.update(label="Analysis failed", state="error")
            progress_bar.empty()
            st.error(str(error))
        except Exception as error:
            status.update(label="Analysis failed", state="error")
            progress_bar.empty()
            st.error(f"Something went wrong: {error}")


def show_pdf_statistics(uploaded_file) -> None:
    """Display PDF file and indexing statistics in a clean grid."""
    indexing = st.session_state.indexing_stats or {}
    details = st.session_state.pdf_details or {}
    char_count = details.get("char_count", 0)

    chars_indexed = indexing.get("chars_indexed")
    if chars_indexed is None:
        chars_indexed = get_text_processing_stats(char_count)["chars_indexed"]
    chars_display = f"{chars_indexed:,}" if isinstance(chars_indexed, int) else "—"

    chunks_display = (
        str(indexing["chunks_created"])
        if "chunks_created" in indexing
        else "—"
    )

    st.markdown("#### 📊 PDF Statistics")

    col1, col2, col3, col4, col5 = st.columns(5)

    stats = [
        ("📄 File Name", uploaded_file.name),
        ("💾 File Size", format_file_size(uploaded_file.size)),
        ("📑 Pages", str(details.get("page_count", "—"))),
        ("🔤 Characters Indexed", chars_display),
        ("🧩 Chunks Created", chunks_display),
    ]

    for column, (label, value) in zip([col1, col2, col3, col4, col5], stats):
        with column:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">{label}</div>
                    <div class="stat-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    show_character_processing_stats(char_count)


def show_analysis_results(analysis: dict) -> None:
    """Display all Version 2 analysis sections inside expanders."""
    st.markdown("---")

    header_col, download_col = st.columns([4, 1])
    with header_col:
        st.subheader("📊 Analysis Results")
    with download_col:
        paper_name = st.session_state.paper_name or "paper.pdf"
        export_filename = f"analysis_{sanitize_paper_filename(paper_name)}.txt"
        st.download_button(
            label="📥 Download Analysis",
            data=format_analysis_export(analysis, paper_name),
            file_name=export_filename,
            mime="text/plain",
            use_container_width=True,
            help="Download all analysis sections as a text file.",
        )

    difficulty = analysis.get("difficulty_rating", "N/A")
    st.markdown(
        f'<div class="difficulty-badge">'
        f"Difficulty Rating: {difficulty} / 10"
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.expander("📋 Executive Summary", expanded=True):
        st.write(analysis.get("executive_summary", "Not available."))

    with st.expander("💡 Key Takeaways"):
        takeaways = analysis.get("key_takeaways", [])
        for takeaway in takeaways:
            st.markdown(f"- {takeaway}")

    with st.expander("🎓 Beginner Explanation (ELI15)"):
        st.write(analysis.get("beginner_explanation", "Not available."))

    with st.expander("❓ Quiz Questions (5)"):
        quiz = analysis.get("quiz_questions", [])
        for index, item in enumerate(quiz, start=1):
            question = item.get("question", "No question provided.")
            answer = item.get("answer", "No answer provided.")
            with st.expander(f"Question {index}: {question}"):
                st.success(f"**Answer:** {answer}")

    with st.expander("💼 Technical Interview Questions (5)"):
        interview = analysis.get("interview_questions", [])
        for index, question in enumerate(interview, start=1):
            st.markdown(f"{index}. {question}")

    with st.expander("🔮 Future Research Ideas (5)"):
        ideas = analysis.get("future_research_ideas", [])
        for index, idea in enumerate(ideas, start=1):
            st.markdown(f"{index}. {idea}")


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


# ---------------------------------------------------------------------------
# Main page — title, upload, analysis, and chat.
# ---------------------------------------------------------------------------
st.markdown('<p class="hero-title">🔬 ScholarAI Research Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">'
    "Upload a research paper PDF to get an AI-powered analysis, "
    "then chat with your paper using smart search (RAG) — all powered by Google Gemini."
    "</p>",
    unsafe_allow_html=True,
)

show_metrics_row()
st.divider()

uploaded_file = st.file_uploader(
    "📎 Choose a PDF file",
    type="pdf",
    help="Upload a research paper in PDF format to get started.",
)

if uploaded_file is not None:
    # If the user uploads a different file, clear old results and chat history.
    if st.session_state.paper_name != uploaded_file.name:
        reset_paper_state()
        st.session_state.paper_name = uploaded_file.name
        st.session_state.pdf_details = extract_pdf_details(uploaded_file)

    st.success("✅ PDF uploaded successfully!")

    show_pdf_statistics(uploaded_file)

    char_count = st.session_state.pdf_details.get("char_count", 0)
    show_large_pdf_warning(char_count)

    st.markdown("---")

    if st.button("🚀 Analyze Paper", type="primary", use_container_width=True):
        run_paper_analysis(uploaded_file, char_count)

    # Show saved analysis after reruns (for example, after chat messages).
    if st.session_state.analysis is not None:
        show_analysis_results(st.session_state.analysis)

elif st.session_state.vector_store is not None:
    st.info(
        f"📄 **{st.session_state.paper_name}** is still loaded. "
        "You can keep chatting below without re-uploading the PDF."
    )

    if st.session_state.pdf_details is not None:
        char_count = st.session_state.pdf_details.get("char_count", 0)
        show_character_processing_stats(char_count)
        show_large_pdf_warning(char_count)

    if st.session_state.analysis is not None:
        show_analysis_results(st.session_state.analysis)

else:
    st.info(
        "👋 Welcome! Upload a PDF research paper above to begin. "
        "You'll get a full AI analysis and a chat interface to ask questions."
    )

# Chat stays available as long as the PDF index is in session_state.
if st.session_state.vector_store is not None:
    show_chat_with_paper()
