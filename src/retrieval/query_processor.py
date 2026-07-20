"""Query processing module for ScholarAI.

Performs query expansion and decomposition using Gemini 2.5 Flash.
Exposes a single `process_query` function to transform user queries before retrieval.
"""

import json
from typing import List

import google.generativeai as genai

from src.utils.gemini import get_generative_model


def process_query(query: str) -> List[str]:
    """Processes the user query using Gemini to expand or decompose it.

    This function classifies the query and either:
      - Expands a simple query with related terms and synonyms.
      - Decomposes a complex query into 2-3 focused sub-queries.

    The returned list will always have the original query at index 0, followed
    by up to 2 unique processed queries in order of importance.

    Args:
        query: Original user search query.

    Returns:
        List of 1 to 3 search query strings.
    """
    if not query or not query.strip():
        return []

    original_query = query.strip()

    # --- Prompt and Structured JSON Output Rationale ---
    # Prompt Design: We instruct the LLM to classify the query as simple or complex
    # and perform only one action (expansion or decomposition) accordingly.
    # Structured JSON: Ensuring a fixed JSON output format prevents parsing issues
    # and enables robust fallback checks.
    # Low Temperature: A temperature of 0.1 restricts randomness, generating
    # highly relevant and focused search terms rather than creative descriptions.
    prompt = (
        "You are an expert search query optimizer for a scientific research paper assistant.\n"
        "Your task is to analyze the user's input search query and decide whether it is simple or complex, "
        "then generate optimized queries to improve hybrid (sparse/dense) RAG retrieval.\n\n"
        "Instructions:\n"
        "1. Classification:\n"
        "   - If the query is simple, short, or containing abbreviations, lightly expand it by generating "
        "     1 or 2 focused variations. Add synonyms, expand abbreviations, or insert domain-specific terminology "
        "     without changing the original user's intent.\n"
        "   - If the query is complex, multi-part, or asks multiple questions, decompose it into 2 to 3 "
        "     focused, simpler sub-queries.\n"
        "   - Do not perform both expansion and decomposition on the same query unless clearly beneficial.\n"
        "2. Output format:\n"
        "   - You must return a JSON object containing a single key 'queries' containing a list of strings "
        "     representing the processed queries.\n"
        "   - The list should be ordered from most important/useful to least important/useful.\n"
        "   - Do not include the original query in the 'queries' list.\n"
        "3. Output constraints:\n"
        "   - Return ONLY valid JSON matching the schema below. No markdown wrappers, no backticks.\n\n"
        f"Original User Query: {original_query}\n\n"
        "JSON Response Schema:\n"
        "{\n"
        '  "queries": ["most useful query variation/sub-query", "next most useful query variation/sub-query"]\n'
        "}"
    )

    try:
        model = get_generative_model("gemini-2.5-flash")

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )

        # Parse and sanitize LLM output
        data = json.loads(response.text)
        if not isinstance(data, dict) or "queries" not in data:
            return [original_query]

        llm_queries = data["queries"]

        # Graceful fallback: empty list or invalid format
        if not isinstance(llm_queries, list) or not llm_queries:
            return [original_query]

        # Clean and filter queries
        cleaned_queries = []
        for q in llm_queries:
            if isinstance(q, (str, int, float)):
                q_str = str(q).strip()
                if q_str:
                    cleaned_queries.append(q_str)

        # Graceful fallback: more than 3 queries returned
        if len(cleaned_queries) > 3:
            return [original_query]

        # Deduplicate while preserving order, starting with original_query
        seen = {original_query}
        result = [original_query]

        for q in cleaned_queries:
            if q not in seen:
                seen.add(q)
                result.append(q)

        # Limit output to a maximum of 3 queries total
        return result[:3]

    except Exception:
        # Graceful fallback: on any error (network, API, parsing), return original query
        return [original_query]
