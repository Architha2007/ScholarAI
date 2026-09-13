import logging

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config.settings import TOP_K_CHUNKS
from src.utils.gemini import call_gemini_with_retry, get_gemini_api_key
from src.utils.validation import validate_query

logger = logging.getLogger(__name__)


def _get_api_key() -> str:
    """Read the Gemini API key from Streamlit secrets or .env."""
    return get_gemini_api_key()


def _format_context_chunks(documents) -> str:
    """Turn retrieved LangChain Document objects into one prompt string.

    Each chunk is labeled so Gemini knows where the information came from.
    """
    formatted_chunks = []

    for document in documents:
        chunk_id = document.metadata.get("chunk_id", "?")
        formatted_chunks.append(
            f"[Chunk {chunk_id}]\n{document.page_content}"
        )

    return "\n\n".join(formatted_chunks)


def ask_paper_question(vector_store: FAISS, question: str) -> dict:
    """Answer a question using retrieved paper chunks and Gemini.

    Returns a dictionary with:
    - answer: Gemini's response
    - source_chunks: list of chunk dicts shown in the UI

    Raises:
        ValueError: If the query is invalid or empty.
        RuntimeError: If vector store or LLM API fails.
    """
    cleaned_question = validate_query(question)

    if vector_store is None:
        logger.warning("No vector store provided to ask_paper_question.")
        return {
            "answer": "No paper database is loaded. Please upload a PDF.",
            "source_chunks": [],
        }

    # Step 1: find the most relevant chunks in FAISS.
    try:
        relevant_docs = vector_store.similarity_search(
            cleaned_question, k=TOP_K_CHUNKS
        )
    except Exception as exc:
        logger.error(f"FAISS search failed during QA: {exc}")
        return {
            "answer": "Failed to search the paper index. Please try again.",
            "source_chunks": [],
        }

    if not relevant_docs:
        return {
            "answer": (
                "I could not find relevant information in this paper "
                "to answer your question."
            ),
            "source_chunks": [],
        }

    # Step 2: build a prompt that includes only retrieved context.
    context = _format_context_chunks(relevant_docs)

    prompt = (
        "You are a helpful research assistant. "
        "Answer the user's question using ONLY the paper excerpts below. "
        "If the excerpts do not contain enough information, say so clearly. "
        "Keep the answer clear and beginner-friendly.\n\n"
        f"Paper excerpts:\n{context}\n\n"
        f"Question: {cleaned_question}\n\n"
        "Answer:"
    )

    # Step 3: ask Gemini 2.5 Flash to generate the final answer with retry.
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=_get_api_key(),
        temperature=0.2,
    )

    def _invoke():
        return llm.invoke(prompt)

    response = call_gemini_with_retry(_invoke)
    answer = response.content

    # Step 4: prepare source chunks for display in Streamlit.
    source_chunks = []
    for document in relevant_docs:
        source_chunks.append(
            {
                "chunk_id": document.metadata.get("chunk_id", "?"),
                "text": document.page_content,
            }
        )

    return {
        "answer": answer,
        "source_chunks": source_chunks,
    }

