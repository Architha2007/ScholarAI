"""Text formatting utilities for the Streamlit UI."""


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
