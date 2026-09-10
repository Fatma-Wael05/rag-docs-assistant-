"""
Loads the embedding model and persisted Chroma vector store, and exposes a
retrieve() function used by the /query route.

The embedder and collection are loaded lazily but cached (via lru_cache) so
they're only ever loaded once per process -- app/main.py's lifespan calls
these at startup so the first request isn't the one paying the load cost.
"""

from functools import lru_cache

from app.core.config import settings


@lru_cache
def get_embedder():
    # Imported lazily so this module can be imported (e.g. by tests that
    # mock this function out) without requiring sentence-transformers to
    # be installed or its model downloaded.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(settings.embedding_model_name)


@lru_cache
def get_collection():
    import chromadb
    client = chromadb.PersistentClient(path=settings.vector_store_dir)
    return client.get_collection(settings.chroma_collection_name)


def retrieve(question: str, k: int | None = None) -> list[dict]:
    """Embed the question and return the top-k most similar chunks from the
    persisted vector store, each with its source file, text, and distance."""
    k = k or settings.retrieval_top_k

    embedder = get_embedder()
    collection = get_collection()

    query_embedding = embedder.encode([question])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=k)

    retrieved = []
    for chunk_id, text, meta, distance in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        retrieved.append({
            "chunk_id": chunk_id,
            "source": meta["source"],
            "text": text,
            "distance": distance,
        })
    return retrieved
