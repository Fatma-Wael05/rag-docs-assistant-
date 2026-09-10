import logging

from fastapi import APIRouter

from app.schemas.query import QueryRequest, QueryResponse
from app.services.generation import generate_answer

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    logger.info("Received query: %s", request.question)
    result = generate_answer(request.question)
    return QueryResponse(answer=result["answer"], sources=result["sources"])
