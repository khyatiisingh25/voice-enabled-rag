\# End-to-End RAG Quality Analysis



\## Objective



Compare the existing production retrieval configuration before and after the

top\_k change:



\- Baseline: top\_k=3

\- Production candidate: top\_k=10



The experiment evaluates the existing 100-query benchmark using the current

RAG answer-generation and validation pipeline.



Production code was not modified as part of this evaluation.



\## Overall Results



| Metric | top\_k=3 | top\_k=10 | Change |

|---|---:|---:|---:|

| Queries | 100 | 100 | 0 |

| Mean correctness | 0.2821 | 0.2684 | -0.0137 |

| Grounded rate | 0.83 | 0.87 | +0.04 |

| Hallucination rate | 0.04 | 0.04 | 0.00 |

| Relevant source rate | 0.67 | 0.98 | +0.31 |

| Non-refusal rate | 0.87 | 0.89 | +0.02 |

| Mean latency | 0.948 ms | 1.962 ms | +1.014 ms |

| Median latency | 1.085 ms | 1.910 ms | +0.825 ms |

| P95 latency | 1.445 ms | 2.515 ms | +1.070 ms |



\## Findings



\### 1. Answer correctness



Mean correctness decreased from 0.2821 to 0.2684.



This indicates that increasing retrieval depth does not automatically

improve final answer quality.



The current deterministic answer-selection logic can select a less useful

sentence when more retrieved candidates are available.



\### 2. Groundedness



Grounded rate improved from 83% to 87%.



This indicates that top\_k=10 generally provides more context that can support

the generated answer.



\### 3. Citation / source quality



Relevant source rate increased substantially from 67% to 98%.



This is the strongest positive result from the experiment.



The relevant source is much more likely to be present in the retrieved

candidate set when top\_k=10 is used.



However, having the relevant source available does not guarantee that the

answer selector chooses it.



\### 4. Hallucination



Hallucination rate remained unchanged at 4%.



Therefore, top\_k=10 did not introduce a measurable increase in hallucination

rate in this benchmark.



\### 5. Latency



Mean latency increased from approximately 0.95 ms to 1.96 ms.



P95 latency increased from approximately 1.45 ms to 2.52 ms.



The absolute latency remains low, but retrieval depth approximately doubles

the measured end-to-end benchmark latency.



\### 6. Regressions



There were 7 queries where top\_k=10 produced a lower correctness score than

top\_k=3:



\- Query 509 — 1 US dollar equals how many euros

\- Query 943 — 2007 VA disability compensation rates

\- Query 1137 — 24 carat gold price

\- Query 1697 — 90 ounces of water equals gallon

\- Query 2057 — Arlena name meaning

\- Query 3934 — How much does a professional Drummer charge

\- Query 7090 — What does the Bible say about us failing



All seven regressions were classified as correctness regressions.



The regression examples indicate that the retrieved context remains grounded,

but the deterministic answer-selection stage may select a weaker or less

direct sentence from the larger candidate set.



\## Conclusion



top\_k=10 improves retrieval coverage significantly:



\- Relevant source rate: 67% -> 98%

\- Grounded rate: 83% -> 87%

\- Hallucination rate: unchanged at 4%



However, final answer correctness decreased:



\- 0.2821 -> 0.2684



There were also 7 observed answer-quality regressions.



Therefore, top\_k=10 should not be considered a complete end-to-end quality

improvement by itself.



The evidence supports retaining the increased retrieval depth only if the

answer-selection/ranking stage is subsequently improved and validated.



The next improvement should focus on selecting the best answer sentence from

the expanded candidate set rather than increasing retrieval depth further.



No production changes were made during this experiment.

