# Production RAG Performance Analysis

## Scope

This analysis measures the current production `/query` RAG path without making production-code changes.

Current production path:

Query
→ Query Embedding
→ FAISS Retrieval
→ Deterministic Grounded Answer
→ API Response

MLX/Gemini generation is not part of the current production `/query` path.

## End-to-End API Latency

Five representative queries were tested through the FastAPI `/query` endpoint.

| Metric | Result |
|---|---:|
| P50 | 24.84 ms |
| P70 | 25.02 ms |
| P100 | 83.76 ms |
| Minimum | 11.52 ms |
| Maximum | 83.76 ms |
| HTTP failures | 0 |

All five requests returned HTTP 200.

The first request was the slowest at 83.76 ms, while subsequent requests were between 11.52 ms and 35.95 ms.

## Stage-Level Latency

The production RAG stages were measured sequentially for the same set of representative queries.

| Stage | P50 | P70 | P100 |
|---|---:|---:|---:|
| Query Embedding | 23.69 ms | 23.69 ms | 37.84 ms |
| FAISS Retrieval | 0.04 ms | 0.04 ms | 5.54 ms |
| Deterministic Fallback | 0.05 ms | 0.05 ms | 0.25 ms |
| RAG Computation Total | 23.77 ms | 23.77 ms | 43.63 ms |

## Bottleneck

Query embedding is the dominant computational stage.

At P50:

- Embedding: 23.69 ms
- Retrieval: 0.04 ms
- Deterministic fallback: 0.05 ms

FAISS retrieval and deterministic fallback contribute negligible latency compared with query embedding.

Therefore, optimization effort should focus on query embedding rather than FAISS retrieval or deterministic fallback.

## 5-Second API Timeout

The API has a configured pipeline timeout of 5 seconds.

The existing timeout test passed:

```text
1 passed, 7 deselected
