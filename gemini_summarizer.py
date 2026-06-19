"""Send paper text to Gemini and get a full research analysis back."""

import json
import os

import google.generativeai as genai
from dotenv import load_dotenv

from rag_store import MAX_ANALYSIS_CHARS

load_dotenv()


def analyze_research_paper(text: str) -> dict:
    """
    Analyze a research paper in a single Gemini call.

    Returns a dictionary with executive summary, takeaways, quiz questions,
    interview questions, future research ideas, beginner explanation,
    and difficulty rating.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Add it to your .env file."
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    if len(text) > MAX_ANALYSIS_CHARS:
        text = text[:MAX_ANALYSIS_CHARS]

    prompt = (
        "You are an expert AI research assistant. "
        "Read the research paper below and return a JSON object with exactly "
        "these keys:\n\n"
        "{\n"
        '  "executive_summary": "A concise 3-5 sentence overview of the paper",\n'
        '  "key_takeaways": ["takeaway 1", "takeaway 2", "takeaway 3", '
        '"takeaway 4", "takeaway 5"],\n'
        '  "beginner_explanation": "Explain the paper like the reader is 15 '
        'years old. Use simple words and relatable examples.",\n'
        '  "quiz_questions": [\n'
        '    {"question": "...", "answer": "..."},\n'
        '    {"question": "...", "answer": "..."},\n'
        '    {"question": "...", "answer": "..."},\n'
        '    {"question": "...", "answer": "..."},\n'
        '    {"question": "...", "answer": "..."}\n'
        "  ],\n"
        '  "interview_questions": [\n'
        '    "technical interview question 1",\n'
        '    "technical interview question 2",\n'
        '    "technical interview question 3",\n'
        '    "technical interview question 4",\n'
        '    "technical interview question 5"\n'
        "  ],\n"
        '  "future_research_ideas": [\n'
        '    "future research direction or improvement 1",\n'
        '    "future research direction or improvement 2",\n'
        '    "future research direction or improvement 3",\n'
        '    "future research direction or improvement 4",\n'
        '    "future research direction or improvement 5"\n'
        "  ],\n"
        '  "difficulty_rating": 7\n'
        "}\n\n"
        "Rules:\n"
        "- key_takeaways must have exactly 5 bullet points.\n"
        "- quiz_questions must have exactly 5 items with question and answer.\n"
        "- interview_questions must have exactly 5 technical questions "
        "suitable for a job interview.\n"
        "- future_research_ideas must have exactly 5 items describing "
        "possible future research directions or improvements based on the paper.\n"
        "- difficulty_rating must be an integer from 1 (very easy) to 10 "
        "(very advanced).\n"
        "- Return only valid JSON, no markdown or extra text.\n\n"
        f"Paper text:\n\n{text}"
    )

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
        ),
    )

    return json.loads(response.text)
