"""
Streamlit chat UI for the RAG document assistant.
Talks to the FastAPI backend via api_client.py -- never hard-codes the
backend URL, reads it from the API_BASE_URL environment variable instead.
"""

import streamlit as st
from dotenv import load_dotenv

from api_client import BackendError, API_BASE_URL, ask_question, check_health

load_dotenv()

st.set_page_config(page_title="RAG Document Assistant", page_icon="📚")
st.title("📚 RAG Document Assistant")
st.caption(
    "Ask a question about LangChain, LangGraph, CrewAI, or LlamaIndex. "
    "Answers are grounded in the project's document corpus, with sources cited."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Friendly startup check so a down backend fails loudly and clearly,
# rather than as a confusing error on the first question.
if not check_health():
    st.error(
        f"⚠️ Can't reach the backend at `{API_BASE_URL}`. "
        "Make sure it's running (`uvicorn app.main:app --reload` in the backend/ folder), "
        "then refresh this page."
    )

# Replay conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"- `{source}`")

question = st.chat_input("Ask a question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ask_question(question)
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander("Sources"):
                        for source in result["sources"]:
                            st.markdown(f"- `{source}`")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })
            except BackendError as e:
                error_message = f"⚠️ {e}"
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                })
