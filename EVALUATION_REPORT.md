# Lab 7 — Integration, Service, Dashboard, and Regression Gate

## 1. What the system does

Lab 7 integrates the retrieval, RAG, evaluation, and guardrail work from Labs 3–6 into a service.

The service builds the retrieval pipeline once at startup using the Lab 3 selected configuration: Markdown-aware chunking with chunk size 400 and dense retrieval. The Lab 4 RAG pipeline is then used for grounded answer generation and citation validation, while the Lab 6 guardrails protect tool-use and untrusted retrieved content.

The service exposes the following endpoints:

- `POST /ask` — accepts a question and returns a grounded answer, citations, latency, cost, cache status, and trace ID.
- `POST /ask/stream` — provides an SSE response path and reports TTFT and total response latency.
- `GET /health` — reports service health and cache/pipeline status.
- `GET /metrics` — reports cost, cache statistics, latency percentiles, model calls, and latency breakdown information.

The service also implements an exact response cache and a semantic cache. Exact cache lookup is performed first; semantic lookup uses an embedding similarity threshold of 0.85.

## 2. Evaluation results

The regression gate evaluates eight metrics using the configured thresholds in `labs/lab7/thresholds.yml`.

| Metric | Measured value | Threshold | Result |
|---|---:|---:|---|
| Correctness | 0.7625 | ≥ 0.75 | PASS |
| Faithfulness | 0.9556 | ≥ 0.90 | PASS |
| Citation validity | 1.0000 | ≥ 0.98 | PASS |
| Refusal recall | 1.0000 | ≥ 0.80 | PASS |
| Refusal precision | 0.7143 | ≥ 0.75 | FAIL |
| Hit@5 | 0.9762 | ≥ 0.85 | PASS |
| Cost/query | $0.0000 | ≤ $0.010 | PASS |
| p95 latency | 1.602 ms | ≤ 6000 ms | PASS |

The selected Lab 3 `markdown-400 dense` configuration has Hit@5 = 0.9762.

Overall, seven of the eight configured regression thresholds pass. The only failing metric is refusal precision.

## 3. Where the system fails

The main measured failure is refusal precision.

The Lab 4 evaluation contains seven total refusals. Five of these correspond to unanswerable questions, giving:

`5 / 7 = 0.7143`

The configured minimum is 0.75. Therefore the regression gate correctly exits with a non-zero status.

Refusal recall is 1.0000, so the system successfully refuses the unanswerable cases in the evaluation set. The problem is that it also refuses some answerable cases, reducing precision.

This failure was deliberately left visible rather than changing the threshold to force a green result. The gate therefore demonstrates that it can detect a quality regression or threshold violation.

## 4. Cost

The captured Lab 7 metrics contain three requests. All three were exact-cache hits, with zero model calls and a measured cost per query of $0.0000.

This result demonstrates the low cost of the cached path, but it should not be interpreted as the cost of an uncached model-generated request. The captured metrics contain no uncached model calls, so the uncached cost was not measured in this particular artifact.

## 5. Speed and latency

The captured Lab 7 metrics contain three cached requests:

- p50 latency: 1.425 ms
- p95 latency: 1.602 ms
- p99 latency: 1.617 ms

These cached measurements are well below the configured p95 target of 6000 ms.

However, these numbers represent the cached path. Earlier uncached tests showed substantially higher latency because retrieval and generation were executed. Therefore the cached result should not be presented as evidence that the uncached p95 target has been comprehensively demonstrated.

The service also records a latency breakdown for retrieval, reranking, generation, validation, and repair generation when an uncached request is processed.

## 6. Unsafe use and limitations

The system treats retrieved document content as untrusted data rather than instructions. Lab 6 adds prompt-injection detection, output controls, privilege restrictions, confirmation requirements for privileged tools, and budget/termination controls.

Important limitations remain:

1. Refusal precision is below the configured regression threshold.
2. The current captured Lab 7 performance artifact is dominated by exact-cache hits.
3. Uncached latency therefore needs additional representative measurement before claiming the 6-second p95 target has been comprehensively demonstrated.
4. The streaming endpoint provides an application-level streaming interface; the current implementation does not provide true token-by-token model streaming because the underlying RAG generation path completes before the response event is emitted.
5. Semantic-cache behavior depends on the selected similarity threshold. A higher threshold reduces incorrect semantic matches but also reduces semantic-cache hits.

## 7. Next three actions

### 1. Improve refusal precision

Investigate the answerable questions that were incorrectly refused. The objective is to reduce false-positive refusals while preserving the current refusal recall of 1.0000.

### 2. Measure and optimize uncached latency

Run a representative set of uncached questions and measure retrieval, generation, validation, and total latency separately. Use these measurements to identify the largest contributor to the current latency and work toward the 6000 ms p95 target.

### 3. Keep the regression gate enforced in CI

Run `labs/lab7/gate.py` under `AIP_OFFLINE=1` in GitHub Actions. The current refusal-precision violation should remain visible so that CI demonstrates a non-zero exit when a configured quality threshold is breached.

## Conclusion

Lab 7 integrates the Labs 3–6 pipeline into a service with caching, observability, streaming, and automated regression checks. The current gate passes correctness, faithfulness, citation validity, refusal recall, Hit@5, cost, and the captured cached latency requirement. Refusal precision remains below its configured threshold, and the gate correctly reports the violation.

The resulting system is therefore instrumented for regression testing, but the current evaluation identifies refusal precision and representative uncached latency as the main areas requiring further work.
