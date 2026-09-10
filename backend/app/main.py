import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.query import router as query_router
from app.core.config import settings
from app.services.retrieval import get_collection, get_embedder
from app.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the embedding model and vector store once at startup, not on
    # every request. Both are cached (lru_cache) so this also warms the
    # cache that app/services/retrieval.retrieve() reuses.
    logger.info("Loading embedding model: %s", settings.embedding_model_name)
    get_embedder()
    logger.info("Loading vector store from: %s", settings.vector_store_dir)
    collection = get_collection()
    logger.info("Vector store ready: %d chunks in collection '%s'",
                collection.count(), settings.chroma_collection_name)
    yield
    logger.info("Shutting down.")


app = FastAPI(title="RAG Document Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)
