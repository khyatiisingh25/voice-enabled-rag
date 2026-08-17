import asyncio

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.core.exceptions import (
    PipelineTimeoutError,
    PipelineUnavailableError,
)
from app.pipeline import pipeline


PIPELINE_TIMEOUT_SECONDS = 5.0


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