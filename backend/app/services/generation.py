"""
Builds the grounded prompt from retrieved chunks and calls the local Ollama
LLM to generate an answer. Ported directly from the retrieval + prompting
logic developed and tested in notebooks/rag_pipeline.ipynb (Phase 2.4).
"""

import ollama

from app.core.config import settings
from app.services.retrieval import retrieve


def build_prompt(question: str, retrieved: list[dict]) -> str:
    context_blocks = []
    for i, r in enumerate(retrieved, 1):
        context_blocks.append(f'[{i}] Source: {r["source"]}\n{r["text"]}')
    context = "\n\n".join(context_blocks)

    prompt = f"""You are a documentation assistant. Answer the question using ONLY the context below. \
If the context does not contain enough information to answer, say so explicitly -- do not use \
outside knowledge. Cite sources using the [n] markers matching the context blocks.

Context:
{context}

Question: {question}

Answer (with [n] citations):"""
    return prompt


def generate_answer(question: str, k: int | None = None) -> dict:
    """Full retrieval + generation: retrieve top-k chunks, build the grounded
    prompt, call the local Ollama LLM, and return the answer alongside the
    sources that were actually retrieved."""
    retrieved = retrieve(question, k=k)
    prompt = build_prompt(question, retrieved)

    response = ollama.generate(model=settings.ollama_model, prompt=prompt)

    return {
        "answer": response["response"].strip(),
        "sources": [r["source"] for r in retrieved],
    }
