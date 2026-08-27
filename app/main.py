import asyncio
import uuid

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.core.exceptions import (
    PipelineTimeoutError,
    PipelineUnavailableError,
)
from app.core.logging import configure_logging, get_logger
from app.pipeline import pipeline
from app.Voice.stt import transcribe


PIPELINE_TIMEOUT_SECONDS = 5.0


configure_logging()
logger = get_logger("zero-signal")


app = FastAPI(
    title="Zero Signal Voice-Enabled RAG",
    version="0.2.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://zero-signal-frontend-production.up.railway.app",
        "https://voice-enabled-rag.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

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


class VoiceQueryResponse(BaseModel):
    transcript: str
    answer: str
    sources: list
    grounded: bool


# ============================================================
# REQUEST LOGGING
# ============================================================

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


# ============================================================
# RAG PIPELINE
# ============================================================

async def execute_pipeline(query: str) -> dict:
    """
    Execute the RAG pipeline with an API-level timeout.
    """

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                pipeline.query,
                query,
            ),
            timeout=PIPELINE_TIMEOUT_SECONDS,
        )

    except asyncio.TimeoutError as exc:
        raise PipelineTimeoutError(
            "Pipeline execution exceeded the allowed timeout."
        ) from exc


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok"}


# ============================================================
# VOICE QUERY
# ============================================================

@app.post(
    "/voice/query",
    response_model=VoiceQueryResponse,
)
async def voice_query(
    audio: UploadFile = File(...),
):
    try:
        import tempfile
        from pathlib import Path

        # ----------------------------------------------------
        # Save uploaded audio temporarily
        # ----------------------------------------------------

        suffix = Path(
            audio.filename or ""
        ).suffix or ".wav"

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp_file:

            temp_path = Path(temp_file.name)

            temp_file.write(
                await audio.read()
            )

        try:
            # ------------------------------------------------
            # Speech → Text
            # ------------------------------------------------

            transcript = await asyncio.to_thread(
                transcribe,
                str(temp_path),
            )

            transcript = transcript.strip()

            logger.info(
                "voice_transcription_success transcript_length=%s",
                len(transcript),
            )

            if not transcript:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Speech-to-text returned "
                        "an empty transcript."
                    ),
                )

            # ------------------------------------------------
            # Transcript → RAG
            # ------------------------------------------------

            rag_result = await execute_pipeline(
                transcript
            )

            # ------------------------------------------------
            # Return BOTH transcript + RAG response
            # ------------------------------------------------

            return {
                "transcript": transcript,
                "answer": rag_result.get(
                    "answer",
                    "",
                ),
                "sources": rag_result.get(
                    "sources",
                    [],
                ),
                "grounded": rag_result.get(
                    "grounded",
                    False,
                ),
            }

        finally:
            # ------------------------------------------------
            # Delete temporary audio file
            # ------------------------------------------------

            temp_path.unlink(
                missing_ok=True
            )

    except HTTPException:
        raise

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
        logger.exception(
            "voice_query_failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Voice query processing failed.",
        ) from exc


# ============================================================
# TEXT QUERY
# ============================================================

@app.post(
    "/query",
    response_model=QueryResponse,
)
async def query(
    request: QueryRequest,
):
    try:
        return await execute_pipeline(
            request.query
        )

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
        logger.exception(
            "query_failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Pipeline execution failed.",
        ) from exc 