"""
Answer user questions about a paper using RAG (Retrieval Augmented Generation).

This file handles the "question answering" side of RAG:
1. Search the FAISS database for chunks similar to the user's question
2. Send those chunks + the question to Gemini 2.5 Flash
3. Return Gemini's answer along with the source chunks used
"""

import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# How many of the most relevant chunks to retrieve for each question.
TOP_K_CHUNKS = 4


def _get_api_key() -> str:
    """Read the Gemini API key from the .env file."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Add it to your .env file."
        )
    return api_key


def _format_context_chunks(documents) -> str:
    """
    Turn retrieved LangChain Document objects into one prompt string.

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
    """
    Answer a question using retrieved paper chunks and Gemini.

    Returns a dictionary with:
    - answer: Gemini's response
    - source_chunks: list of chunk dicts shown in the UI
    """
    # Step 1: find the most relevant chunks in FAISS.
    relevant_docs = vector_store.similarity_search(question, k=TOP_K_CHUNKS)

    if not relevant_docs:
        return {
            "answer": "I could not find relevant information in this paper.",
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
        f"Question: {question}\n\n"
        "Answer:"
    )

    # Step 3: ask Gemini 2.5 Flash to generate the final answer.
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=_get_api_key(),
        temperature=0.2,
    )

    response = llm.invoke(prompt)
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
