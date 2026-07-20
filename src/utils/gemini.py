"""Gemini API configuration and client helpers."""

import os
import google.generativeai as genai


def get_gemini_api_key() -> str:
    """Retrieves the Gemini API key from Streamlit secrets or environment variables.

    Raises:
        ValueError: If the API key is not found in either location.
    """
    try:
        import streamlit as st

        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in Streamlit secrets or environment variables."
        )
    return api_key


def get_generative_model(
    model_name: str = "gemini-2.5-flash"
) -> genai.GenerativeModel:
    """Configures the Gemini API client and returns a GenerativeModel instance.

    Args:
        model_name: Name of the Gemini model to instantiate.

    Returns:
        GenerativeModel: Configured model instance.
    """
    api_key = get_gemini_api_key()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)
