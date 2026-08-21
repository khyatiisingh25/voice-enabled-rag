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

- `top_k = 10`
- `score_threshold = 0.20`

Low-relevance results are rejected.

If no relevant document is found, the system returns:

```text
grounded: false

## Evaluation Results

### Retrieval Quality

Retrieval was evaluated using the fixed MSMARCO evaluation subset with 100 queries across five chunking strategies.

| Chunking Strategy | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| 500/50 | 0.38 | 0.67 | 0.78 | 0.5298 |
| 300/50 | 0.35 | 0.61 | 0.76 | 0.5000 |
| 800/100 | 0.38 | 0.68 | 0.79 | 0.5342 |
| Sentence | 0.38 | 0.66 | 0.78 | 0.5357 |
| Metadata | 0.36 | 0.67 | 0.78 | 0.5208 |

The 800/100 strategy achieved the highest Recall@3 and Recall@5, while sentence-based chunking achieved the highest MRR. No single strategy dominated all retrieval metrics.

### Latency Comparison

The latency-critical production RAG path measures query embedding, FAISS retrieval, and deterministic grounded answer selection after the query text is available.

| Path | P50 | P70 | P100 |
|---|---:|---:|---:|
| Deterministic fallback | 22.39 ms | 22.39 ms | 32.91 ms |
| MLX generation | 167.90 ms | 167.90 ms | 172.16 ms |
| Gemini baseline* | Not available | Not available | Not available |

The deterministic fallback had the lowest measured latency among the paths with completed percentile measurements.

The Gemini baseline benchmark produced successful individual requests in approximately the 1–1.8 second range, but the full benchmark was interrupted by the Gemini API free-tier rate limit. Therefore, full-run Gemini P50/P70/P100 values are not reported.

\* Gemini latency is reported separately from the latency-critical deterministic production boundary.

### Grounding Validation

The grounding validation benchmark evaluated three representative questions. Two answers passed validation and one was rejected because the answer copied the question. This benchmark result is not intended as a production-wide grounding accuracy estimate.

- Validation pass rate: **2/3 (66.7%)**
- Total validation time: **0.1033 ms**

The rejected case was classified as `question_copy`, demonstrating that the validator rejects an answer that does not provide grounded content.
