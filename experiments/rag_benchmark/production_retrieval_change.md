# Production Retrieval Improvement

## Objective

Validate whether the retrieval-quality analysis can be safely integrated into the production RAG pipeline.

The primary candidate evaluated was increasing production retrieval depth from `top_k=3` to `top_k=10`.

No chunking strategy or similarity threshold was changed.

---

## Baseline Production Configuration

Current production retrieval configuration before the change:

- Chunking: existing production `chunk_text()`
- Retriever: FAISS `IndexFlatIP`
- Embeddings: `all-MiniLM-L6-v2`
- Similarity threshold: `0.20`
- Production `top_k`: `3`

Baseline retrieval metrics:

| Metric | Before |
|---|---:|
| Recall@1 | 0.38 |
| Recall@3 | 0.68 |
| Recall@5 | 0.68 |
| MRR | 0.5067 |

---

## Controlled Top-k Experiment

The production chunking function was kept unchanged.

The following configurations were evaluated on the same 100-query evaluation set:

| top_k | Recall@1 | Recall@3 | Recall@5 | MRR |
|---:|---:|---:|---:|---:|
| 3 | 0.38 | 0.68 | 0.68 | 0.5067 |
| 5 | 0.38 | 0.67 | 0.78 | 0.5298 |
| 10 | 0.38 | 0.67 | 0.78 | 0.5580 |

### Observations

Increasing `top_k` did not improve Recall@1.

Recall@3 decreased slightly from `0.68` to `0.67`.

However, Recall@5 improved from `0.68` to `0.78`, and MRR improved from `0.5067` to `0.5580`.

This indicates that a larger retrieval candidate depth improves deeper retrieval coverage and reciprocal rank, but does not improve the highest-ranked result.

---

## Rejected Retrieval Improvements

Additional controlled experiments were performed before modifying production.

### Source-diversity selection

The source-diversity experiment produced:

| Metric | Baseline | Source-diverse |
|---|---:|---:|
| Recall@1 | 0.38 | 0.38 |
| Recall@3 | 0.69 | 0.68 |
| Recall@5 | 0.69 | 0.68 |
| MRR | 0.5100 | 0.5067 |

The candidate configuration performed slightly worse and was therefore not integrated into production.

### Lexical reranking

The lexical reranking experiment tested multiple semantic/lexical weighting configurations.

Best candidate by Recall@1:

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| Semantic only | 0.38 | 0.68 | 0.68 | 0.5067 |
| 90% semantic / 10% lexical | 0.39 | 0.65 | 0.65 | 0.5033 |
| 80% semantic / 20% lexical | 0.38 | 0.65 | 0.65 | 0.4983 |
| 70% semantic / 30% lexical | 0.35 | 0.66 | 0.66 | 0.4850 |

Although the 90/10 configuration increased Recall@1 by 0.01, it reduced Recall@3, Recall@5 and MRR.

Therefore lexical reranking was not integrated.

---

## Production Change

Based on the controlled evidence, production retrieval depth was changed:

```python
top_k=3