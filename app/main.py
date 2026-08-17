import asyncio
import uuid

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from app.core.exceptions import (
    PipelineTimeoutError,
    PipelineUnavailableError,
)
from app.core.logging import configure_logging, get_logger
from app.pipeline import pipeline


PIPELINE_TIMEOUT_SECONDS = 5.0


configure_logging()
logger = get_logger("zero-signal")


app = FastAPI(
    title="Zero Signal Voice-Enabled RAG",
    version="0.2.0",
)


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )


class QueryResponse(BaseModel):
    answer: str
    sources: list
    grounded: bool


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())

    logger.info(
        "request_started request_id=%s method=%s path=%s",
        request_id,
        request.method,
        request.url.path,
    )

    try:
        response = await call_next(request)

        logger.info(
            "request_completed request_id=%s status=%s",
            request_id,
            response.status_code,
        )

        response.headers["X-Request-ID"] = request_id

        return response

    except Exception:
        logger.exception(
            "request_failed request_id=%s method=%s path=%s",
            request_id,
            request.method,
            request.url.path,
        )
        raise


async def execute_pipeline(query: str) -> dict:
    """
    Execute the pipeline with a hard API-level timeout.

    The actual downstream RAG/LLM calls should also have
    their own provider-level timeouts when integrated.
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(pipeline.query, query),
            timeout=PIPELINE_TIMEOUT_SECONDS,
        )

    except asyncio.TimeoutError as exc:
        raise PipelineTimeoutError(
            "Pipeline execution exceeded the allowed timeout."
        ) from exc


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        return await execute_pipeline(request.query)

    except PipelineTimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=str(exc),
        ) from exc

    except PipelineUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Pipeline execution failed.",
        ) from exc