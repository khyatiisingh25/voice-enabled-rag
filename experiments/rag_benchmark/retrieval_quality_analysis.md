# Retrieval Quality Analysis & Improvement

## 1. Objective

This analysis evaluates retrieval quality across the five available chunking strategies and controlled retrieval configurations.

The purpose is to identify the main causes of missed retrievals and determine whether chunking strategy, `top_k`, or similarity threshold can improve retrieval quality.

No production retrieval pipeline changes were made as part of this analysis.

---

## 2. Baseline

Evaluation dataset:

- MSMARCO evaluation subset
- Queries evaluated: 100
- Embedding model: `all-MiniLM-L6-v2`
- Retriever: FAISS `IndexFlatIP`
- Evaluation metrics: Recall@1, Recall@3, Recall@5, MRR

Existing baseline results:

| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| 500/50 | 0.38 | 0.67 | 0.78 | 0.5298 |
| 300/50 | 0.35 | 0.61 | 0.76 | 0.5000 |
| 800/100 | 0.38 | 0.68 | 0.79 | 0.5342 |
| sentence | 0.38 | 0.66 | 0.78 | 0.5357 |
| metadata | 0.36 | 0.67 | 0.78 | 0.5208 |

---

## 3. Chunking Strategy Comparison

The `800/100` strategy produced the strongest overall recall:

- Recall@1: 0.38
- Recall@3: 0.68
- Recall@5: 0.79
- MRR: 0.5342

The `300/50` strategy performed worst:

- Recall@1: 0.35
- Recall@3: 0.61
- Recall@5: 0.76
- MRR: 0.5000

Therefore, smaller 300-character chunks did not improve retrieval on this evaluation set.

The `800/100` configuration is the strongest chunking candidate among the five tested strategies.

---

## 4. Top-k Comparison

Controlled experiments were performed with:

- top_k = 3
- top_k = 5
- top_k = 10

For `800/100`:

| top_k | Recall@1 | Recall@3 | Recall@5 | MRR |
|---:|---:|---:|---:|---:|
| 3 | 0.38 | 0.69 | 0.69 | 0.5100 |
| 5 | 0.38 | 0.68 | 0.79 | 0.5342 |
| 10 | 0.38 | 0.68 | 0.79 | 0.5647* |

`top_k=10` produced the highest observed MRR in the controlled experiment.

However, increasing `top_k` does not improve the ranking of already retrieved documents. It provides a larger candidate set and can recover relevant passages that occur below rank 5.

Therefore, `top_k=10` should be considered a candidate configuration rather than an immediate production change.

---

## 5. Similarity Threshold Comparison

Thresholds tested:

- 0.10
- 0.20
- 0.30

Across the tested strategies and top-k values, Recall@1, Recall@3, and Recall@5 remained effectively unchanged across these thresholds.

This indicates that similarity threshold tuning is not the primary retrieval bottleneck for this evaluation set.

The missed-query analysis also supports this conclusion.

For example, the query:

"2 liters is how many ounces"

had several incorrect results with very high similarity scores:

- 0.8479
- 0.8129
- 0.8022
- 0.7437
- 0.7090

The correct passage was not present in the top five.

Therefore, simply lowering or increasing the threshold is unlikely to solve the main retrieval failures.

---

## 6. Missed Retrieval Analysis

Configuration analyzed:

- Strategy: 800/100
- top_k: 5
- threshold: 0.20
- Queries evaluated: 100
- Missed queries: 21
- Miss rate: 21%

Therefore:

- 79/100 queries retrieved a relevant passage within the top 5.
- 21/100 queries did not retrieve a relevant passage within the top 5.

### Observed failure patterns

#### 6.1 Semantic ranking failures

Several missed queries had incorrect passages with very high similarity scores.

Examples include:

- "2 liters is how many ounces"
- "Ginevra name meaning"
- "Pelvic inflammatory disease..."
- "What is the meaning of the name Mitchell"

This indicates that the embedding retriever can assign high similarity to semantically related but non-relevant passages.

This is a ranking-quality issue rather than a simple threshold issue.

#### 6.2 Short or ambiguous queries

Examples include:

- "+what is centure"
- "5k how long"
- "SOAPS definition"

Short queries provide limited semantic context and are more difficult for the embedding model to distinguish accurately.

#### 6.3 Similarity-score separation

In several misses, incorrect passages received scores substantially above the threshold.

For example, the `Ginevra name meaning` query produced incorrect top results with scores above 0.89.

This demonstrates that the issue is not that the relevant result is simply being removed by the 0.20 threshold. The incorrect documents are being ranked above the relevant document.

#### 6.4 Duplicate passage/chunk retrieval

For the query:

"What does the name Roman mean"

the top two results were both associated with `passage_0`.

This suggests that multiple chunks from the same original passage can occupy multiple retrieval positions.

A future improvement could investigate passage-level deduplication or reranking so that the candidate list contains more diverse source passages.

This has not been implemented in production.

---

## 7. Main Causes of Missed Retrievals

Based on the controlled experiments and missed-query analysis, the primary causes are:

1. Semantic ranking errors from the embedding model.
2. Short or ambiguous queries.
3. High similarity scores for non-relevant passages.
4. Multiple chunks from the same source passage competing for top positions.
5. Some relevant passages ranking below the selected top-k cutoff.

Similarity threshold does not appear to be the primary bottleneck.

---

## 8. Recommendation

### Recommended chunking candidate

`800/100`

Reason:

It produced the best overall Recall@5 among the tested chunking strategies:

- Recall@5 = 0.79

### Recommended retrieval candidate

`top_k=10` should be evaluated as a candidate configuration because it produced the highest observed MRR in the controlled experiment:

- MRR = 0.5647

However, this should not be applied to the production pipeline yet.

### Similarity threshold

No threshold change is recommended based on the current evidence.

The tested thresholds of 0.10, 0.20 and 0.30 produced effectively unchanged recall metrics.

### Future improvement areas

The next retrieval-quality experiments should investigate:

- passage-level deduplication
- reranking of retrieved candidates
- query normalization for short/noisy queries
- stronger embedding models
- larger candidate retrieval followed by reranking

These are recommendations for future controlled experiments, not production changes.

---

## 9. Production Change Status

No production retrieval pipeline changes were made.

The following benchmark/evaluation artifacts remain unchanged:

- existing retrieval benchmark numbers
- existing production retriever
- existing deterministic fallback
- existing chunker

The analysis was performed using separate benchmark artifacts.

---

## 10. Final Conclusion

The controlled evaluation shows that `800/100` is the strongest chunking strategy among the five tested strategies.

Increasing `top_k` from 5 to 10 improves the available candidate depth and produced the highest observed MRR, but does not improve Recall@1.

Similarity threshold changes from 0.10 to 0.30 did not materially improve recall.

The dominant retrieval problem is therefore semantic ranking quality rather than threshold filtering.

The recommended next investigation is candidate reranking/deduplication and improved handling of short or ambiguous queries.

No production changes should be made until these recommendations are validated through another controlled benchmark.