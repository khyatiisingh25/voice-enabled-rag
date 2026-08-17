from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.pipeline import pipeline


app = FastAPI(
    title="Zero Signal Voice-Enabled RAG",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    answer: str
    sources: list
    grounded: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        return pipeline.query(request.query)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Pipeline execution failed",
        ) from exc