#!/usr/bin/env python3
"""Lab 3 — retrieval sweeps.

The scaffolding (corpus loading, metric computation, table printing) is
written for you. The sweeps are yours.

    python labs/lab3/search.py --baseline
    python labs/lab3/search.py --sweep chunking
    python labs/lab3/search.py --sweep retrieval
    python labs/lab3/search.py --sweep rerank
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.chunking import STRATEGIES, Chunk  # noqa: E402
from aip.evals import retrieval_metrics  # noqa: E402
from aip.retrieval import (
    DenseRetriever,
    Bm25Retriever,
    HybridRetriever,
    ChromaRetriever,
    CrossEncoderReranker,
    LLMReranker,
)

CORPUS_DIR = ROOT / "data/corpus"
GOLDEN = ROOT / "data/eval/rag_golden.jsonl"


# ---------------------------------------------------------------------------
# scaffolding (provided)
# ---------------------------------------------------------------------------
def load_corpus() -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(CORPUS_DIR.glob("*.md"))}


def load_questions(include_unanswerable: bool = False) -> list[dict]:
    rows = [json.loads(l) for l in GOLDEN.open(encoding="utf-8")]
    if include_unanswerable:
        return rows
    # THREE questions (Q36, Q38, Q39) have no relevant document, so recall and
    # nDCG are undefined for them -- you cannot rank correctly against an empty
    # relevant set. Dropping them leaves n = 42.
    #
    # Do not confuse that with the FIVE questions of kind 'unanswerable'
    # (Q36-Q40): two of those do keep relevant documents, because part of what
    # they ask is supported. All five are measured properly in Lab 4, as
    # refusal precision and recall.
    #
    # Excluding the three is correct -- but say so in your report rather than
    # letting an unexplained n = 42 pass for a stated 45.
    return [r for r in rows if r["relevant_docs"]]


def build_chunks(corpus: dict[str, str], strategy: str = "sliding",
                 size: int = 800, **kw) -> list[Chunk]:
    fn = STRATEGIES[strategy]
    out: list[Chunk] = []
    for doc_id, text in corpus.items():
        try:
            out.extend(fn(text, doc_id, size=size, **kw))
        except TypeError:                       # chunker without that kwarg
            out.extend(fn(text, doc_id, size=size))
    return out


def evaluate(retriever: Retriever, questions: list[dict], k: int = 10,
             reranker=None, final_k: int = 5) -> dict:
    """Run every question, return aggregate metrics + per-kind breakdown."""
    agg: dict[str, list[float]] = defaultdict(list)
    by_kind: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    latencies: list[float] = []
    per_q: dict[str, float] = {}
    per_q_mrr: dict[str, float] = {}

    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], k=k)
        if reranker is not None:
            hits = reranker.rerank(q["question"], hits, k=final_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        # A document counts as retrieved at rank r if any of its chunks does.
        seen, ranked = set(), []
        for h in hits:
            if h.doc_id not in seen:
                seen.add(h.doc_id)
                ranked.append(h.doc_id)

        m = retrieval_metrics(ranked, q["relevant_docs"], ks=(1, 3, 5, 10))
        per_q[q["id"]] = m["hit_rate@5"]
        per_q_mrr[q["id"]] = m["mrr"]
        for key, val in m.items():
            agg[key].append(val)
            by_kind[q["kind"]][key].append(val)

    out = {k2: statistics.fmean(v) for k2, v in agg.items()}
    out["latency_p50_ms"] = statistics.median(latencies)
    out["latency_p95_ms"] = sorted(latencies)[int(0.95 * (len(latencies) - 1))]
    out["_by_kind"] = {kind: {k2: statistics.fmean(v) for k2, v in d.items()}
                       for kind, d in by_kind.items()}
    out["_per_question"] = per_q            # hit_rate@5 -- saturated, see kind_table
    out["_per_question_mrr"] = per_q_mrr    # use this one for Part B
    out["_kind_n"] = {kind: len(d["mrr"]) for kind, d in by_kind.items()}
    return out


def table(rows: dict[str, dict], cols: tuple[str, ...] =
          ("hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10",
           "latency_p95_ms")) -> str:
    name_w = max(len(n) for n in rows) + 2
    head = f"{'config':<{name_w}}" + "".join(f"{c:>15}" for c in cols)
    lines = [head, "-" * len(head)]
    for name, m in rows.items():
        lines.append(f"{name:<{name_w}}" + "".join(f"{m.get(c, 0):>15.4f}" for c in cols))
    return "\n".join(lines)


def kind_table(metrics: dict, col: str = "hit_rate@5") -> str:
    """Break a result down by question kind.

    NOTE the default column. `hit_rate@5` is saturated on this corpus -- every
    retriever scores 0.93-0.98 -- so this table will look flat and tell you
    nothing. Pass col='mrr' or col='ndcg@10' for Part B. The default is left
    saturated on purpose.
    """
    bk, counts = metrics["_by_kind"], metrics.get("_kind_n", {})
    w = max(len(k) for k in bk) + 2
    lines = [f"{'kind':<{w}}{col:>12}{'n':>6}", "-" * (w + 18)]
    for kind, m in sorted(bk.items()):
        lines.append(f"{kind:<{w}}{m.get(col, 0):>12.4f}{counts.get(kind, 0):>6}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# sweeps (yours)
# ---------------------------------------------------------------------------
def sweep_baseline() -> None:
    corpus, questions = load_corpus(), load_questions()
    chunks = build_chunks(corpus, "sliding", 800, overlap=150)
    print(f"corpus: {len(corpus)} docs -> {len(chunks)} chunks "
          f"(mean {statistics.fmean(len(c) for c in chunks):.0f} chars)")
    r = DenseRetriever(chunks)
    m = evaluate(r, questions)
    print(table({"baseline sliding-800 dense": m}))
    print()
    print(kind_table(m))
    print("\nWrite these numbers down before you change anything.")


def sweep_chunking():
    corpus, questions = load_corpus(), load_questions()

    # A1: Compare chunking strategies at size 800
    strategies = ["fixed", "sliding", "recursive", "markdown"]

    a1_results = {}

    for strategy in strategies:
        chunks = build_chunks(
            corpus,
            strategy,
            800,
            overlap=150
        )

        r = DenseRetriever(chunks)
        m = evaluate(r, questions)

        a1_results[f"{strategy}-800 dense"] = m

    print("\nA1: Chunking strategy comparison")
    print(table(a1_results))

    # A2: Compare chunk sizes using Markdown chunking
    a2_results = {}

    for size in [400, 800, 1600]:
        chunks = build_chunks(
            corpus,
            "markdown",
            size,
            overlap=150
        )

        r = DenseRetriever(chunks)
        m = evaluate(r, questions)

        a2_results[f"markdown-{size} dense"] = m

    print("\nA2: Markdown chunk-size comparison")
    print(table(a2_results))

    # A3: Compare Markdown with and without heading-path prefix
    a3_results = {}

    for include_heading_path in [False, True]:
        chunks = build_chunks(
            corpus,
            "markdown",
            400,
            include_heading_path=include_heading_path
        )

        r = DenseRetriever(chunks)
        m = evaluate(r, questions)

        label = "with-heading-path" if include_heading_path else "without-heading-path"
        a3_results[f"markdown-400 {label}"] = m

    print("\nA3: Markdown heading-path comparison")
    print(table(a3_results))


def sweep_retrieval():
    corpus, questions = load_corpus(), load_questions()

    # Use the best chunking configuration from Part A:
    # Markdown chunking, size 400, heading path included
    chunks = build_chunks(
        corpus,
        "markdown",
        400,
        include_heading_path=True
    )

    # ============================================================
    # B1: Retrieval comparison
    # ============================================================
    print("\nB1: Retrieval comparison")

    results = {}

    # Dense retrieval
    dense = DenseRetriever(chunks)
    results["markdown-400 dense"] = evaluate(
        dense,
        questions
    )

    # BM25 retrieval
    bm25 = Bm25Retriever(chunks)
    results["markdown-400 bm25"] = evaluate(
        bm25,
        questions
    )

    # Hybrid retrieval = Dense + BM25
    hybrid = HybridRetriever([
        DenseRetriever(chunks),
        Bm25Retriever(chunks)
    ])

    results["markdown-400 hybrid"] = evaluate(
        hybrid,
        questions
    )

    print(table(results))


    # ============================================================
    # B2: Question-kind breakdown
    # ============================================================
    print("\nB2: Question-kind breakdown")

    retrievers = {
        "dense": DenseRetriever(chunks),

        "bm25": Bm25Retriever(chunks),

        "hybrid": HybridRetriever([
            DenseRetriever(chunks),
            Bm25Retriever(chunks)
        ]),
    }

    for name, retriever in retrievers.items():

        m = evaluate(
            retriever,
            questions
        )

        print(f"\n{name.upper()}")

        # Use MRR because Hit@5 is often saturated
        print(
            kind_table(
                m,
                col="mrr"
            )
        )

        # Special questions required by the lab
        print(
            f"Q41 MRR = "
            f"{m['_per_question_mrr'].get('Q41', 'N/A')}"
        )

        print(
            f"Q44 MRR = "
            f"{m['_per_question_mrr'].get('Q44', 'N/A')}"
        )


    # ============================================================
    # B3: RRF k sweep
    # ============================================================
    print("\nB3: RRF k sweep")

    rrf_results = {}

    for k in [10, 30, 60, 100]:

        retriever = HybridRetriever(
            [
                DenseRetriever(chunks),
                Bm25Retriever(chunks)
            ],
            rrf_k=k
        )

        m = evaluate(
            retriever,
            questions
        )

        rrf_results[
            f"hybrid-rrf-k{k}"
        ] = m

    print(table(rrf_results))


    # ============================================================
    # B4: Unequal fusion weights
    # ============================================================
    print("\nB4: Unequal fusion weights")

    weight_results = {}

    for weights in [
        [1.0, 1.0],
        [2.0, 1.0]
    ]:

        retriever = HybridRetriever(
            [
                DenseRetriever(chunks),
                Bm25Retriever(chunks)
            ],
            rrf_k=10,
            weights=weights
        )

        m = evaluate(
            retriever,
            questions
        )

        weight_results[
            f"weights-{weights[0]}-{weights[1]}"
        ] = m

    print(table(weight_results))


def sweep_rerank() -> None:
    """C1-C4: Reranking experiments."""

    corpus, questions = load_corpus(), load_questions()

    # Best chunking configuration from Part A
    chunks = build_chunks(
        corpus,
        "markdown",
        400,
        include_heading_path=True
    )

    # Use dense retrieval as the first-stage retriever.
    # Retrieve 30 candidates, then rerank to the final 5.
    retriever = DenseRetriever(chunks)

    # ============================================================
    # C1: Cross-encoder reranking
    # ============================================================
    print("\nC1: Cross-encoder reranking")

    print("Loading cross-encoder...")
    cross_encoder = CrossEncoderReranker()

    # Baseline: dense retrieval, k=30
    baseline = evaluate(
        retriever,
        questions,
        k=30
    )

    # Cross-encoder: retrieve 30, rerank to 5
    cross_metrics = evaluate(
        retriever,
        questions,
        k=30,
        reranker=cross_encoder,
        final_k=5
    )

    print("\nBaseline: Dense retrieval")
    print(table({
        "dense-k30": baseline
    }))

    print("\nCross-encoder reranked")
    print(table({
        "dense-k30-crossencoder-5": cross_metrics
    }))

    # Delta
    print("\nC1: Cross-encoder delta")

    print(
        f"nDCG@10: "
        f"{baseline['ndcg@10']:.4f} -> "
        f"{cross_metrics['ndcg@10']:.4f} "
        f"(delta = "
        f"{cross_metrics['ndcg@10'] - baseline['ndcg@10']:+.4f})"
    )

    print(
        f"Hit@1:   "
        f"{baseline['hit_rate@1']:.4f} -> "
        f"{cross_metrics['hit_rate@1']:.4f} "
        f"(delta = "
        f"{cross_metrics['hit_rate@1'] - baseline['hit_rate@1']:+.4f})"
    )

    print(
        f"Recall@5: "
        f"{baseline['recall@5']:.4f} -> "
        f"{cross_metrics['recall@5']:.4f} "
        f"(delta = "
        f"{cross_metrics['recall@5'] - baseline['recall@5']:+.4f})"
    )

    print(
        f"p95 latency: "
        f"{baseline['latency_p95_ms']:.2f} ms -> "
        f"{cross_metrics['latency_p95_ms']:.2f} ms "
        f"(added = "
        f"{cross_metrics['latency_p95_ms'] - baseline['latency_p95_ms']:+.2f} ms)"
    )


    # ============================================================
    # C2: LLM reranking
    # ============================================================
    print("\nC2: LLM reranking")

    llm_reranker = LLMReranker()

    llm_metrics = evaluate(
        retriever,
        questions,
        k=30,
        reranker=llm_reranker,
        final_k=5
    )

    print(table({
        "dense-k30-llm-5": llm_metrics
    }))


    # ============================================================
    # C3: Deployment decision table
    # ============================================================
    print("\nC3: Deployment decision table")

    print(
        "config                         "
        "nDCG@10    hit@1     p95_ms"
    )

    print(
        f"dense baseline                "
        f"{baseline['ndcg@10']:.4f}     "
        f"{baseline['hit_rate@1']:.4f}     "
        f"{baseline['latency_p95_ms']:.2f}"
    )

    print(
        f"cross-encoder                 "
        f"{cross_metrics['ndcg@10']:.4f}     "
        f"{cross_metrics['hit_rate@1']:.4f}     "
        f"{cross_metrics['latency_p95_ms']:.2f}"
    )

    print(
        f"LLM reranker                  "
        f"{llm_metrics['ndcg@10']:.4f}     "
        f"{llm_metrics['hit_rate@1']:.4f}     "
        f"{llm_metrics['latency_p95_ms']:.2f}"
    )


    # ============================================================
    # C4: Find questions where reranking made things worse
    # ============================================================
    print("\nC4: Questions where cross-encoder reranking made things worse")

    baseline_mrr = baseline["_per_question_mrr"]
    cross_mrr = cross_metrics["_per_question_mrr"]

    worse = []

    for qid in baseline_mrr:
        before = baseline_mrr.get(qid, 0.0)
        after = cross_mrr.get(qid, 0.0)

        if after < before:
            worse.append(
                (qid, before, after, after - before)
            )

    if worse:
        for qid, before, after, delta in worse:
            print(
                f"{qid}: "
                f"before={before:.4f}, "
                f"after={after:.4f}, "
                f"delta={delta:+.4f}"
            )
    else:
        print("No questions were made worse by cross-encoder reranking.")

def sweep_index() -> None:
    """D1-D3: Index comparison, scaling, and metadata filtering."""

    import time
    from pathlib import Path

    corpus, questions = load_corpus(), load_questions()

    # ============================================================
    # Best configuration from Part A
    # Markdown chunking, size 400, heading path enabled
    # ============================================================
    chunks = build_chunks(
        corpus,
        "markdown",
        400,
        include_heading_path=True
    )

    # ============================================================
    # D1: Exact NumPy vs Chroma HNSW
    # ============================================================
    print("\nD1: Exact NumPy vs Chroma HNSW")

    # ------------------------------------------------------------
    # Exact NumPy dense retrieval
    # ------------------------------------------------------------
    dense = DenseRetriever(chunks)

    dense_metrics = evaluate(
        dense,
        questions
    )

    # ------------------------------------------------------------
    # Chroma HNSW
    # ------------------------------------------------------------
    chroma = ChromaRetriever(
        chunks,
        path=".chroma_lab3",
        collection="lab3_d1",
        reset=True
    )

    chroma_metrics = evaluate(
        chroma,
        questions
    )

    print("\nExact NumPy")
    print(table({
        "dense-exact": dense_metrics
    }))

    print("\nChroma HNSW")
    print(table({
        "chroma-hnsw": chroma_metrics
    }))

    print(
        f"\nRecall@5 gap: "
        f"{chroma_metrics['recall@5'] - dense_metrics['recall@5']:+.4f}"
    )

    print(
        f"nDCG@10 gap: "
        f"{chroma_metrics['ndcg@10'] - dense_metrics['ndcg@10']:+.4f}"
    )

    # ============================================================
    # D2: Scaling experiment
    # ============================================================
    print("\nD2: Scaling experiment")

    root = Path(__file__).resolve().parents[2]
    scaled_dir = root / "data" / "corpus_scaled"

    # Use one fixed query for a fair latency comparison.
    timing_query = questions[0]["question"]

    def time_retriever(
        retriever,
        query: str,
        repeats: int = 5
    ) -> float:
        """Return median search latency in milliseconds."""

        times = []

        # Warm-up search
        retriever.search(query, k=10)

        # Timed searches
        for _ in range(repeats):
            start = time.perf_counter()

            retriever.search(
                query,
                k=10
            )

            elapsed = (
                time.perf_counter() - start
            ) * 1000

            times.append(elapsed)

        times.sort()

        # Median
        return times[len(times) // 2]

    def run_scale(
        label: str,
        corpus_for_index: dict
    ):
        """Build exact and HNSW indexes and compare latency."""

        # Build chunks
        scale_chunks = build_chunks(
            corpus_for_index,
            "markdown",
            400,
            include_heading_path=True
        )

        print(f"\n{label}")
        print(f"Chunks: {len(scale_chunks)}")

        # --------------------------------------------------------
        # Exact NumPy
        # --------------------------------------------------------
        exact = DenseRetriever(
            scale_chunks
        )

        exact_ms = time_retriever(
            exact,
            timing_query
        )

        # --------------------------------------------------------
        # Chroma HNSW
        # --------------------------------------------------------
        collection_name = (
            "lab3_scale_"
            + label.lower()
            .replace(" ", "_")
            .replace("~", "")
        )

        hnsw = ChromaRetriever(
            scale_chunks,
            path=".chroma_lab3",
            collection=collection_name,
            reset=True
        )

        hnsw_ms = time_retriever(
            hnsw,
            timing_query
        )

        ratio = (
            hnsw_ms / exact_ms
            if exact_ms
            else float("inf")
        )

        print(
            f"Exact NumPy: {exact_ms:.3f} ms"
        )

        print(
            f"Chroma HNSW: {hnsw_ms:.3f} ms"
        )

        print(
            f"HNSW / Exact: {ratio:.2f}x"
        )

        return (
            len(scale_chunks),
            exact_ms,
            hnsw_ms
        )

    # Store D2 measurements
    d2_results = []

    # ------------------------------------------------------------
    # D2.1: Original corpus
    # Approximately 160 chunks
    # ------------------------------------------------------------
    d2_results.append(
        run_scale(
            "~160 chunks",
            corpus
        )
    )

    # ------------------------------------------------------------
    # D2.2 and D2.3 require scaled corpus
    # ------------------------------------------------------------
    if not scaled_dir.exists():

        print(
            "\nScaled corpus not found."
        )

        print(
            "Run:"
        )

        print(
            "python scripts/expand_corpus.py --docs 4000"
        )

    else:

        scaled_files = sorted(
            scaled_dir.glob("*.md")
        )

        # --------------------------------------------------------
        # D2.2: Approximately 4k chunks
        #
        # Each filler document produces roughly 10 chunks.
        # 400 filler documents therefore give approximately 4k
        # chunks.
        # --------------------------------------------------------
        small_scaled_files = scaled_files[:400]

        small_corpus = dict(corpus)

        for p in small_scaled_files:

            small_corpus[p.stem] = (
                p.read_text(
                    encoding="utf-8"
                )
            )

        d2_results.append(
            run_scale(
                "~4k chunks",
                small_corpus
            )
        )

        # --------------------------------------------------------
        # D2.3: Approximately 40k chunks
        #
        # Use all 4,000 filler documents.
        # --------------------------------------------------------
        full_scaled_corpus = dict(corpus)

        for p in scaled_files:

            full_scaled_corpus[p.stem] = (
                p.read_text(
                    encoding="utf-8"
                )
            )

        d2_results.append(
            run_scale(
                "~40k chunks",
                full_scaled_corpus
            )
        )

    # ------------------------------------------------------------
    # D2 Summary
    # ------------------------------------------------------------
    print("\nD2 Summary")

    print(
        f"{'Scale':<15}"
        f"{'Chunks':>10}"
        f"{'Exact ms':>15}"
        f"{'HNSW ms':>15}"
        f"{'HNSW/Exact':>15}"
    )

    print("-" * 70)

    for chunks_count, exact_ms, hnsw_ms in d2_results:

        print(
            f"{'':<15}"
            f"{chunks_count:>10}"
            f"{exact_ms:>15.3f}"
            f"{hnsw_ms:>15.3f}"
            f"{hnsw_ms / exact_ms:>14.2f}x"
        )

    # ============================================================
    # D3: Metadata filtering
    # ============================================================
    print("\nD3: Metadata filtering")

    # ------------------------------------------------------------
    # Add status metadata to every chunk
    #
    # Archived documents have "ARCHIVED" in their doc_id.
    # Everything else is treated as current.
    # ------------------------------------------------------------
    for chunk in chunks:

        chunk.meta["status"] = (
            "archived"
            if "ARCHIVED" in chunk.doc_id
            else "current"
        )

    # ------------------------------------------------------------
    # Fresh Chroma collection so metadata is ingested
    # ------------------------------------------------------------
    chroma_meta = ChromaRetriever(
        chunks,
        path=".chroma_lab3",
        collection="lab3_metadata",
        reset=True
    )

    # ------------------------------------------------------------
    # Questions required by the lab
    # ------------------------------------------------------------
    target_ids = {
        "Q29",
        "Q30",
        "Q31"
    }

    target_questions = [
        q
        for q in questions
        if q["id"] in target_ids
    ]

    # ------------------------------------------------------------
    # D3 Before filtering
    # ------------------------------------------------------------
    before = {}

    for q in target_questions:

        hits = chroma_meta.search(
            q["question"],
            k=10
        )

        ranked = []
        seen = set()

        for h in hits:

            if h.doc_id not in seen:

                seen.add(h.doc_id)
                ranked.append(h.doc_id)

        m = retrieval_metrics(
            ranked,
            q["relevant_docs"],
            ks=(1, 5, 10)
        )

        before[q["id"]] = (
            m["hit_rate@1"]
        )

    # ------------------------------------------------------------
    # D3 After filtering
    #
    # Only current policies are allowed.
    # ------------------------------------------------------------
    after = {}

    for q in target_questions:

        hits = chroma_meta.search(
            q["question"],
            k=10,
            where={
                "status": "current"
            }
        )

        ranked = []
        seen = set()

        for h in hits:

            if h.doc_id not in seen:

                seen.add(h.doc_id)
                ranked.append(h.doc_id)

        m = retrieval_metrics(
            ranked,
            q["relevant_docs"],
            ks=(1, 5, 10)
        )

        after[q["id"]] = (
            m["hit_rate@1"]
        )

    # ------------------------------------------------------------
    # Print D3 results
    # ------------------------------------------------------------
    print(
        "\nQ29-Q31 Hit@1 before vs after metadata filtering"
    )

    print(
        f"{'Question':<15}"
        f"{'Before':>12}"
        f"{'After':>12}"
        f"{'Delta':>12}"
    )

    print("-" * 49)

    for qid in [
        "Q29",
        "Q30",
        "Q31"
    ]:

        delta = (
            after[qid]
            - before[qid]
        )

        print(
            f"{qid:<15}"
            f"{before[qid]:>12.4f}"
            f"{after[qid]:>12.4f}"
            f"{delta:>12.4f}"
        )


SWEEPS = {
    "chunking": sweep_chunking,
    "retrieval": sweep_retrieval,
    "rerank": sweep_rerank,
    "index": sweep_index,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--sweep", choices=list(SWEEPS))
    args = ap.parse_args()
    if args.baseline or not args.sweep:
        sweep_baseline()
    if args.sweep:
        SWEEPS[args.sweep]()


if __name__ == "__main__":
    main()
