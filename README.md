# Voice-Enabled RAG

Voice-Enabled RAG system built by Team Zero Signal for Hacker House Goa 2026.

## Overview

The system combines Speech-to-Text (STT), Retrieval-Augmented Generation (RAG), FastAPI, and grounded answer generation to answer questions using information retrieved from project documents.

The latency-critical production RAG path is designed to operate without requiring Gemini generation. Gemini remains available for the generation path and related experiments/benchmarks.

## Architecture

Voice Input

↓

Speech-to-Text (STT)

↓

Clean Transcript

↓

POST /query

↓

Query Embedding

↓

FAISS Retrieval

↓

Deterministic Grounded Answer

↓

answer + sources + grounded

### Latency-Critical Production Boundary

The production RAG latency measurement starts when the query text is available after STT and ends when the grounded RAG answer is produced.

Included in the RAG latency measurement:

- Query embedding generation
- FAISS retrieval
- Deterministic fallback answer selection

Excluded from the RAG latency measurement:

- Audio capture
- Speech-to-Text (STT)
- STT preprocessing
- Network/API transport outside the RAG pipeline
- Model initialization / cold start
- Gemini generation

STT latency is reported separately from production RAG latency.

## RAG Pipeline

Documents are loaded from `data/documents/`, split into chunks, converted into embeddings using Sentence Transformers, and indexed using FAISS.

Retrieval uses:

- `top_k = 3`
- `score_threshold = 0.20`

Low-relevance results are rejected.

If no relevant document is found, the system returns:

```text
grounded: false