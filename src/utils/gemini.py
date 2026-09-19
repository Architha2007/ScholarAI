"""Gemini API configuration and client helpers."""

import logging
import os
import time
from typing import Any, Callable

import google.generativeai as genai

logger = logging.getLogger(__name__)


def get_gemini_api_key() -> str:
    """Retrieves the Gemini API key from Streamlit secrets or env vars.

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
            "GEMINI_API_KEY not found in Streamlit secrets or "
            "environment variables."
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


def call_gemini_with_retry(
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    **kwargs: Any,
) -> Any:
    """Calls a Gemini API function with exponential backoff retries.

    Args:
        fn: Function to call.
        max_retries: Maximum number of attempt retries.
        initial_delay: Delay in seconds before first retry.
        backoff_factor: Multiplier for backoff delay.

    Returns:
        Result from fn.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exception = exc
            err_msg = str(exc)
            logger.warning(
                f"Gemini API call failed (attempt {attempt}/{max_retries}): "
                f"{err_msg[:100]}"
            )
            if attempt == max_retries:
                break
            time.sleep(delay)
            delay *= backoff_factor

    logger.error(
        f"Gemini API retries exhausted after {max_retries} attempts: "
        f"{str(last_exception)[:100]}"
    )
    raise RuntimeError(
        "Gemini API service is currently unavailable or rate-limited. "
        "Please try again in a few moments."
    ) from last_exception
