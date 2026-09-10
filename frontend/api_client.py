"""
Thin wrapper around the FastAPI backend's /query and /health endpoints.
Kept separate from app.py so the Streamlit UI code doesn't need to know
about requests/HTTP details directly.
"""

import os

import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

# Generation can take a while on a local Ollama model -- give it real room
# rather than timing out on a slow first request.
REQUEST_TIMEOUT_SECONDS = 60


class BackendError(Exception):
    """Raised when the backend is unreachable or returns an error."""


def check_health() -> bool:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def ask_question(question: str) -> dict:
    """Call POST /query. Returns {"answer": str, "sources": list[str]}.
    Raises BackendError with a human-readable message on any failure."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": question},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.ConnectionError as e:
        raise BackendError(
            f"Could not reach the backend at {API_BASE_URL}. "
            "Is it running? (uvicorn app.main:app --reload)"
        ) from e
    except requests.Timeout as e:
        raise BackendError(
            "The backend took too long to respond. The local LLM may still be "
            "generating -- try again in a moment."
        ) from e

    if response.status_code == 422:
        raise BackendError("The backend rejected the request (invalid input).")
    if response.status_code >= 500:
        raise BackendError(f"The backend returned a server error ({response.status_code}).")
    if response.status_code != 200:
        raise BackendError(f"Unexpected response from backend: {response.status_code}")

    data = response.json()
    return {"answer": data["answer"], "sources": data["sources"]}
