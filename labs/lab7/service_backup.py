from __future__ import annotations
from fastapi import FastAPI, HTTPException, Request
import sys
import time
import json
from pathlib import Path

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from aip.cost import BudgetExceeded, global_budget
from fastapi import FastAPI, HTTPException

import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from aip import cache, tracing
from aip.chunking import markdown_chunks
from aip.cost import BudgetExceeded, global_budget
from aip.embed import embed, cosine
from aip.retrieval import DenseRetriever

from labs.lab3.search import load_corpus
from labs.lab4.rag import answer_question

from labs.lab4.rag import (
    ANSWER_SYSTEM,
    REFUSAL,
    answer_question,
    validate_answer,
)
from aip.llm import chat_stream
from aip.retrieval import format_context


# ============================================================
# PATH SETUP
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Aurora Policy Assistant",
    version="1.0",
)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    ERROR_STATS["422_validation"] += 1

    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
        },
    )

@app.post("/lab7/semantic-cache/clear")
def clear_semantic_cache():
    """
    Development/evaluation endpoint.
    Clears only the in-memory semantic cache.
    Does not modify the persistent AIP cache.
    """
    count = len(SEMANTIC_CACHE)

    SEMANTIC_CACHE.clear()

    return {
        "status": "cleared",
        "entries_removed": count,
        "semantic_cache_entries": len(SEMANTIC_CACHE),
    }

# ============================================================
# GLOBAL STATE
# ============================================================

_PIPELINE = None
_STARTED = time.time()

import os

SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.85"))


# ============================================================
# CACHE STATISTICS
# ============================================================

CACHE_STATS = {
    "exact_hits": 0,
    "semantic_hits": 0,
    "misses": 0,
    "semantic_candidates": 0,
}

ERROR_STATS = {
    "422_validation": 0,
    "429_budget": 0,
    "503_provider": 0,
    "500_internal": 0,
}

TOOL_STATS = {
    "total": 0,
    "allowed": 0,
    "denied": 0,
}


# ============================================================
# SERVICE-LEVEL LATENCY STATISTICS
# ============================================================

REQUEST_LATENCIES_MS = []

STREAM_TTFT_MS = []

STREAM_TOTAL_LATENCIES_MS = []


# ============================================================
# RAG LATENCY BREAKDOWN
# ============================================================

LATENCY_BREAKDOWN = {
    "retrieval_ms": [],
    "reranking_ms": [],
    "generation_ms": [],
    "validation_ms": [],
    "repair_generation_ms": [],
    "total_ms": [],
}


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class AskRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=1000,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    mode: str = Field(
        default="rag",
        pattern="^(rag|tools)$",
    )


class Citation(BaseModel):
    index: int
    doc_id: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Citation]
    sources: list[Citation] = []
    latency_ms: float
    cost_usd: float
    cached: bool
    trace_id: str
    semantic_similarity: float | None = None
    semantic_match: str | None = None


# ============================================================
# QUESTION NORMALIZATION
# ============================================================

def normalize_question(question: str) -> str:
    """
    Normalize whitespace and case so equivalent questions
    produce the same exact-cache key.
    """

    return " ".join(
        question.strip().split()
    ).lower()


# ============================================================
# EXACT CACHE
# ============================================================

def make_response_cache_key(
    question: str,
    top_k: int,
    mode: str,
) -> str:

    normalized = normalize_question(
        question
    )

    return cache.make_key(
        "lab7.ask",
        {
            "question": normalized,
            "top_k": top_k,
            "mode": mode,
        },
    )


# ============================================================
# SEMANTIC CACHE STORAGE
# ============================================================

SEMANTIC_CACHE: list[dict] = []


def add_semantic_cache_entry(
    question: str,
    top_k: int,
    mode: str,
    response_data: dict,
    question_embedding: np.ndarray,
) -> None:

    normalized = normalize_question(
        question
    )

    SEMANTIC_CACHE.append(
        {
            "question": normalized,
            "top_k": top_k,
            "mode": mode,
            "embedding": np.asarray(
                question_embedding,
                dtype=np.float32,
            ),
            "response": response_data,
        }
    )


def semantic_cache_lookup(
    question: str,
    top_k: int,
    mode: str,
    query_embedding=None,
) -> tuple[dict | None, float | None, str | None]:

    if not SEMANTIC_CACHE:
        return None, None, None

    normalized = normalize_question(
        question
    )

    # Reuse an existing query embedding when one was
    # already computed by the caller.
    if query_embedding is None:
        query_embedding = embed(
            normalized,
            input_type="query",
        )

    best_similarity = -1.0
    best_entry = None

    for entry in SEMANTIC_CACHE:

        if entry["top_k"] != top_k:
            continue

        if entry["mode"] != mode:
            continue

        CACHE_STATS[
            "semantic_candidates"
        ] += 1

        similarity = float(
            cosine(
                query_embedding,
                entry["embedding"],
            )
        )

        if similarity > best_similarity:
            best_similarity = similarity
            best_entry = entry

    if (
        best_entry is not None
        and best_similarity >= SEMANTIC_THRESHOLD
    ):
        return (
            best_entry["response"],
            best_similarity,
            best_entry["question"],
        )

    return (
        None,
        best_similarity
        if best_entry
        else None,
        None,
    )



@app.post("/lab7/semantic-cache/eval")
def semantic_cache_eval(
    req: AskRequest,
    threshold: float | None = None,
):
    """
    Evaluation-only semantic-cache lookup.

    Bypasses the exact cache, does not run RAG,
    and does not modify SEMANTIC_CACHE.

    If threshold is supplied, it is used only for
    this evaluation request. The production
    SEMANTIC_THRESHOLD is not changed.
    """

    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")

    evaluation_threshold = (
        SEMANTIC_THRESHOLD
        if threshold is None
        else threshold
    )

    if not SEMANTIC_CACHE:
        return {
            "cached": False,
            "semantic_similarity": None,
            "semantic_match": None,
            "best_match_similarity": None,
            "entries_checked": 0,
            "threshold_used": evaluation_threshold,
        }

    normalized = normalize_question(req.question)

    query_embedding = embed(
        normalized,
        input_type="query",
    )

    best_similarity = -1.0
    best_entry = None
    candidates = 0

    for entry in SEMANTIC_CACHE:

        if entry["top_k"] != req.top_k:
            continue

        if entry["mode"] != req.mode:
            continue

        candidates += 1

        similarity = float(
            cosine(
                query_embedding,
                entry["embedding"],
            )
        )

        if similarity > best_similarity:
            best_similarity = similarity
            best_entry = entry

    if best_entry is None:
        return {
            "cached": False,
            "semantic_similarity": None,
            "semantic_match": None,
            "best_match_similarity": None,
            "entries_checked": candidates,
            "threshold_used": evaluation_threshold,
        }

    matched = best_similarity >= evaluation_threshold

    return {
        "cached": matched,
        "semantic_similarity": best_similarity,
        "semantic_match": (
            best_entry["question"]
            if matched
            else None
        ),
        "best_match_similarity": best_similarity,
        "entries_checked": candidates,
        "threshold_used": evaluation_threshold,
    }

@app.post("/lab7/semantic-cache/seed")
def seed_semantic_cache():
    """
    Development/evaluation endpoint.

    Seeds the in-memory semantic cache from existing exact-cache
    responses. This avoids running the LLM just to create test
    seed entries.

    The six seed questions are fixed so threshold experiments
    are reproducible.
    """

    SEMANTIC_CACHE.clear()

    seed_questions = [
        "What is the deadline for submitting a reimbursement claim?",
        "How long do I have to report a grievance?",
        "What is the grace period for policy renewal?",
        "What documents are required for KYC verification?",
        "How long does Aurora take to respond to a customer query?",
        "What expenses are covered after hospitalization?",
    ]

    top_k = 5
    mode = "rag"

    entries = 0
    retriever = pipeline()

    for question in seed_questions:
        normalized = normalize_question(question)

        # ----------------------------------------------------
        # 1. Try the existing persistent exact cache
        # ----------------------------------------------------
        cache_key = make_response_cache_key(
            question,
            top_k,
            mode,
        )

        response_data = cache.get(cache_key)

        # ----------------------------------------------------
        # 2. If not already cached, generate the answer
        # ----------------------------------------------------
        if response_data is None:
            result = answer_question(
                question,
                retriever,
                k=10,
                final_k=top_k,
            )

            response_data = {
                "answer": result.text,
                "refused": result.refused,
                "citations": [],
            }

        # ----------------------------------------------------
        # 3. Store using the SAME helper as production /ask
        # ----------------------------------------------------
        question_embedding = embed(normalized)

        add_semantic_cache_entry(
            question=question,
            top_k=top_k,
            mode=mode,
            response_data=response_data,
            question_embedding=question_embedding,
        )

        entries += 1

    return {
        "status": "seeded",
        "entries": entries,
        "semantic_cache_entries": len(SEMANTIC_CACHE),
        "top_k": top_k,
        "mode": mode,
    }
# ============================================================
# PIPELINE
# ============================================================

def pipeline():

    global _PIPELINE

    if _PIPELINE is not None:
        return _PIPELINE

    corpus = load_corpus()

    chunks = [
        chunk
        for doc_id, text in corpus.items()
        for chunk in markdown_chunks(
            text,
            doc_id,
            size=400,
        )
    ]

    retriever = DenseRetriever(
        chunks
    )

    _PIPELINE = retriever

    print(
        f"Pipeline ready: "
        f"{len(chunks)} chunks, "
        f"dense retrieval, "
        f"markdown-400"
    )

    return _PIPELINE


# ============================================================
# CITATION BUILDER
# ============================================================

def build_citations(hits) -> list[Citation]:

    citations = []

    for i, hit in enumerate(
        hits,
        start=1,
    ):

        chunk = hit.chunk

        excerpt = getattr(
            chunk,
            "text",
            "",
        )

        if not excerpt:
            excerpt = str(chunk)

        doc_id = (
            getattr(
                chunk,
                "doc_id",
                None,
            )
            or getattr(
                hit,
                "source",
                "",
            )
        )

        citations.append(
            Citation(
                index=i,
                doc_id=str(doc_id),
                excerpt=str(excerpt)[
                    :500
                ],
            )
        )

    return citations

def stream_rag_answer(
    question: str,
    retriever,
    final_k: int = 5,
    tier: str = "MAIN",
    query_embedding=None,
):
    """Stream the Lab 4 RAG generation without modifying Lab 4.

    Retrieval and prompt construction intentionally mirror
    labs.lab4.rag.answer_question().
    """

    # ---------------------------------------------------------
    # 1. Retrieve
    # ---------------------------------------------------------
    retrieval_start = time.perf_counter()

    hits = retriever.search(
        question,
        k=10,
        query_embedding=query_embedding,
    )

    retrieval_ms = (
        time.perf_counter() - retrieval_start
    ) * 1000


    print(
        f"[PERF] retrieval.search = {retrieval_ms:.2f} ms "
        f"| embedding_reused = {query_embedding is not None}"
    )

    # ---------------------------------------------------------
    # 2. No reranker for the current Lab 7 pipeline.
    #    This matches the normal service path.
    # ---------------------------------------------------------
    hits = hits[:final_k]

    # ---------------------------------------------------------
    # 3. Format context exactly as Lab 4 does.
    # ---------------------------------------------------------
    context = format_context(hits)

    prompt = f"""\
Question:

{question}


Sources:

{context}


Answer the question using only the supplied sources.
"""

    # ---------------------------------------------------------
    # 4. Stream generation.
    # ---------------------------------------------------------
    generation_start = time.perf_counter()

    chunks = []

    for chunk in chat_stream(
        prompt,
        system=ANSWER_SYSTEM,
        tier=tier,
        temperature=0,
        max_tokens=1024,
    ):
        chunks.append(chunk)

        yield {
            "type": "token",
            "text": chunk,
        }

    generation_ms = (
        time.perf_counter() - generation_start
    ) * 1000

    answer = "".join(chunks)

    # ---------------------------------------------------------
    # 5. Validate after the stream has completed.
    #
    # IMPORTANT:
    # validation cannot happen before TTFT because that would
    # require buffering the entire answer again.
    # ---------------------------------------------------------
    validation = validate_answer(
        answer,
        len(hits),
        None,
    )

    refused = answer.strip() == REFUSAL

    total_ms = (
        retrieval_ms
        + generation_ms
    )

    yield {
        "type": "stream_complete",
        "answer": answer,
        "refused": refused,
        "hits": hits,
        "validation": validation,
        "latency_breakdown": {
            "retrieval_ms": retrieval_ms,
            "reranking_ms": 0.0,
            "generation_ms": generation_ms,
            "validation_ms": 0.0,
            "repair_generation_ms": 0.0,
            "total_ms": total_ms,
        },
    }

# ============================================================
# PERCENTILE HELPER
# ============================================================

def percentile(
    values: list[float],
    percentile_value: float,
):

    if not values:
        return None

    values = sorted(values)

    index = (
        (len(values) - 1)
        * (percentile_value / 100)
    )

    lower = int(index)

    upper = min(
        lower + 1,
        len(values) - 1,
    )

    weight = index - lower

    return round(
        values[lower]
        + (
            values[upper]
            - values[lower]
        ) * weight,
        3,
    )


# ============================================================
# LATENCY BREAKDOWN RECORDER
# ============================================================

def record_latency_breakdown(
    breakdown: dict,
) -> None:

    for key in LATENCY_BREAKDOWN:

        if key in breakdown:

            LATENCY_BREAKDOWN[
                key
            ].append(
                float(
                    breakdown[key]
                )
            )


# ============================================================
# SSE EVENT HELPER
# ============================================================

def sse_event(data: dict) -> str:

    return (
        f"data: "
        f"{json.dumps(data)}"
        f"\n\n"
    )


# ============================================================
# ASK ENDPOINT
# ============================================================

@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask(req: AskRequest):

    request_start = time.perf_counter()

    cache_key = make_response_cache_key(
        req.question,
        req.top_k,
        req.mode,
    )

    with tracing.trace(
        "http.ask",
        question=req.question[:120],
        mode=req.mode,
        top_k=req.top_k,
    ) as span:

        # ====================================================
        # 1. EXACT CACHE
        # ====================================================

        cached_response = cache.get(
            cache_key
        )

        if cached_response is not None:

            CACHE_STATS[
                "exact_hits"
            ] += 1

            latency_ms = (
                time.perf_counter()
                - request_start
            ) * 1000

            REQUEST_LATENCIES_MS.append(
                latency_ms
            )

            span["cached"] = True
            span["cache_type"] = "exact"
            span[
                "latency_ms_request"
            ] = latency_ms

            
            return AskResponse(
                answer=cached_response["answer"],
                refused=cached_response["refused"],
                citations=[
                    Citation(**c)
                    for c in cached_response["citations"]
                ],
                sources=[
                    Citation(**c)
                    for c in cached_response["citations"]
                ],
                latency_ms=latency_ms,
                cost_usd=0.0,
                cached=True,
                trace_id=span["span_id"],
            )
    
        # =================================================
        # 2. SEMANTIC CACHE
        # =================================================
        query_embedding = None
        semantic_response = None
        semantic_similarity = None
        semantic_match = None

        try:

            embedding_start = time.perf_counter()

            query_embedding = embed(
                normalize_question(
                    req.question
                ),
                input_type="query",
            )

            embedding_ms = (
                time.perf_counter() - embedding_start
            ) * 1000

            print(
                f"[PERF] stream query embedding = "
                f"{embedding_ms:.2f} ms"
            )

            (
                semantic_response,
                semantic_similarity,
                semantic_match,
            ) = semantic_cache_lookup(
                req.question,
                req.top_k,
                req.mode,
                query_embedding=query_embedding,
            )

        except Exception as exc:

            print(
                f"[PERF] semantic cache error: {exc}"
            )

            semantic_response = None
            semantic_similarity = None
            semantic_match = None


        # ====================================================
        # 3. CACHE MISS
        # ====================================================

        CACHE_STATS[
            "misses"
        ] += 1

        span["cached"] = False
        span["cache_type"] = "miss"

        # ====================================================
        # 4. RAG PIPELINE
        # ====================================================

        retriever = pipeline()

        # ====================================================
        # 5. RUN RAG
        # ====================================================

        budget_before = (
            global_budget()
            .spent_usd
        )

        
        try:
            result = answer_question(
                req.question,
                retriever,
                k=10,
                final_k=req.top_k,
            )

        except BudgetExceeded as exc:
            ERROR_STATS["429_budget"] += 1
            raise HTTPException(
                status_code=429,
                detail=str(exc),
            ) from exc

        except Exception as exc:
            ERROR_STATS["503_provider"] += 1
            raise HTTPException(
                status_code=503,
                detail="upstream model unavailable",
                headers={"Retry-After": "5"},
            ) from exc



        record_latency_breakdown(
            result.latency_breakdown
        )

        span[
            "latency_breakdown"
        ] = result.latency_breakdown

        budget_after = (
            global_budget()
            .spent_usd
        )

        cost_usd = max(
            0.0,
            float(
                budget_after
                - budget_before
            ),
        )

        # ====================================================
        # 6. BUILD RESPONSE
        # ====================================================

        answer = result.text

        refused = result.refused

        hits = result.hits

        citations = build_citations(
            hits
        )

        latency_ms = (
            time.perf_counter()
            - request_start
        ) * 1000

        REQUEST_LATENCIES_MS.append(
            latency_ms
        )

        trace_id = span[
            "span_id"
        ]

        span[
            "cost_usd"
        ] = cost_usd

        span[
            "latency_ms_request"
        ] = latency_ms

        span[
            "refused"
        ] = refused

        span[
            "n_citations"
        ] = len(citations)

        # ====================================================
        # 7. SERIALIZABLE RESPONSE
        # ====================================================

        response_data = {
            "answer": answer,
            "refused": refused,
            "citations": [
                citation.model_dump()
                for citation in citations
            ],
        }

        # ====================================================
        # 8. EXACT CACHE
        # ====================================================

        cache.put(
            cache_key,
            "lab7.ask",
            {
                "question":
                    normalize_question(
                        req.question
                    ),
                "top_k": req.top_k,
                "mode": req.mode,
            },
            response_data,
        )

        # ====================================================
        # 9. SEMANTIC CACHE
        # ====================================================

        try:

            question_embedding = embed(
                normalize_question(
                    req.question
                ),
                input_type="query",
            )

            add_semantic_cache_entry(
                question=req.question,
                top_k=req.top_k,
                mode=req.mode,
                response_data=response_data,
                question_embedding=(
                    question_embedding
                ),
            )

            span[
                "semantic_cache_stored"
            ] = True

        except Exception as exc:

            span[
                "semantic_cache_store_error"
            ] = str(exc)

        # ====================================================
        # 10. RETURN
        # ====================================================

        return AskResponse(
            answer=answer,
            refused=refused,
            citations=citations,
            sources=citations,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            cached=False,
            trace_id=trace_id,
        )


# ============================================================
# STREAMING ENDPOINT
# ============================================================


@app.post(
    "/ask/stream"
)
def ask_stream(
    req: AskRequest,
):
    request_start = time.perf_counter()

    def generate():
        try:

            # =================================================
            # 1. EXACT CACHE
            # =================================================

            cache_key = make_response_cache_key(
                req.question,
                req.top_k,
                req.mode,
            )

            cached_response = cache.get(
                cache_key
            )

            if cached_response is not None:

                CACHE_STATS[
                    "exact_hits"
                ] += 1

                elapsed_ms = (
                    time.perf_counter()
                    - request_start
                ) * 1000

                STREAM_TTFT_MS.append(
                    elapsed_ms
                )

                yield sse_event({
                    "type": "meta",
                    "cached": True,
                    "cache_type": "exact",
                    "ttft_ms": round(
                        elapsed_ms,
                        2,
                    ),
                })

                yield sse_event({
                    "type": "token",
                    "text": cached_response[
                        "answer"
                    ],
                })

                total_latency_ms = (
                    time.perf_counter()
                    - request_start
                ) * 1000

                STREAM_TOTAL_LATENCIES_MS.append(
                    total_latency_ms
                )

                yield sse_event({
                    "type": "done",
                    "answer": cached_response[
                        "answer"
                    ],
                    "refused": cached_response[
                        "refused"
                    ],
                    "citations": cached_response.get(
                        "citations",
                        [],
                    ),
                    "latency_ms": round(
                        total_latency_ms,
                        2,
                    ),
                    "cost_usd": 0.0,
                })

                return

            # =================================================
            # 2. SEMANTIC CACHE
            # =================================================

            query_embedding = None

            try:

                if SEMANTIC_CACHE:
                    embedding_start = time.perf_counter()

                    # Compute the query embedding ONCE.
                    query_embedding = embed(
                        normalize_question(
                            req.question
                        ),
                        input_type="query",
                    )

                    embedding_ms = (time.perf_counter() - embedding_start) * 1000

                    print(
                        f"[PERF] stream query embedding = {embedding_ms:.2f} ms"
                    )

                    (
                        semantic_response,
                        semantic_similarity,
                        semantic_match,
                    ) = semantic_cache_lookup(
                        req.question,
                        req.top_k,
                        req.mode,
                        query_embedding=query_embedding,
                    )

                else:

                    (
                        semantic_response,
                        semantic_similarity,
                        semantic_match,
                    ) = semantic_cache_lookup(
                        req.question,
                        req.top_k,
                        req.mode,
                    )

            except Exception as exc:

                semantic_response = None
                semantic_similarity = None
                semantic_match = None

            if semantic_response is not None:

                CACHE_STATS[
                    "semantic_hits"
                ] += 1

                elapsed_ms = (
                    time.perf_counter()
                    - request_start
                ) * 1000

                STREAM_TTFT_MS.append(
                    elapsed_ms
                )

                yield sse_event({
                    "type": "meta",
                    "cached": True,
                    "cache_type": "semantic",
                    "similarity": semantic_similarity,
                    "matched_question": semantic_match,
                    "ttft_ms": round(
                        elapsed_ms,
                        2,
                    ),
                })

                yield sse_event({
                    "type": "token",
                    "text": semantic_response[
                        "answer"
                    ],
                })

                total_latency_ms = (
                    time.perf_counter()
                    - request_start
                ) * 1000

                STREAM_TOTAL_LATENCIES_MS.append(
                    total_latency_ms
                )

                yield sse_event({
                    "type": "done",
                    "answer": semantic_response[
                        "answer"
                    ],
                    "refused": semantic_response[
                        "refused"
                    ],
                    "citations": semantic_response.get(
                        "citations",
                        [],
                    ),
                    "semantic_similarity":
                        semantic_similarity,
                    "semantic_match":
                        semantic_match,
                    "latency_ms": round(
                        total_latency_ms,
                        2,
                    ),
                    "cost_usd": 0.0,
                })

                return

            # =================================================
            # 3. CACHE MISS
            # =================================================

            CACHE_STATS[
                "misses"
            ] += 1

            # =================================================
            # 4. RUN RAG
            # =================================================

            retriever = pipeline()

            budget_before = (
                global_budget()
                .spent_usd
            )

            # =================================================
            # 5. RUN STREAMING RAG
            # =================================================

            first_token_sent = False
            first_token_ms = None

            answer_parts = []
            final_info = None

            print(
                f"[PERF] before stream_rag_answer: "
                f"query_embedding is None = {query_embedding is None}"
            )

            for event in stream_rag_answer(
                req.question,
                retriever,
                final_k=req.top_k,
                tier="MAIN",
                query_embedding=query_embedding,
            ):

                if event["type"] == "token":

                    chunk = event["text"]

                    if not first_token_sent:

                        first_token_ms = (
                            time.perf_counter()
                            - request_start
                        ) * 1000

                        first_token_sent = True

                        STREAM_TTFT_MS.append(
                            first_token_ms
                        )

                        yield sse_event({
                            "type": "meta",
                            "cached": False,
                            "ttft_ms": round(
                                first_token_ms,
                                2,
                            ),
                            "refused": False,
                        })

                    answer_parts.append(
                        chunk
                    )

                    yield sse_event({
                        "type": "token",
                        "text": chunk,
                    })

                elif event["type"] == "stream_complete":

                    final_info = event

            # =================================================
            # 6. STREAM COMPLETION CHECK
            # =================================================

            if final_info is None:

                raise RuntimeError(
                    "Streaming RAG ended without completion metadata."
                )

            answer = "".join(
                answer_parts
            )

            refused = final_info[
                "refused"
            ]

            hits = final_info[
                "hits"
            ]

            validation = final_info[
                "validation"
            ]

            # =================================================
            # 7. BUILD CITATIONS
            # =================================================

            citations = build_citations(
                hits
            )

            # =================================================
            # 8. COST
            # =================================================

            budget_after = (
                global_budget()
                .spent_usd
            )

            cost_usd = max(
                0.0,
                float(
                    budget_after
                    - budget_before
                ),
            )

            # =================================================
            # 9. LATENCY BREAKDOWN
            # =================================================

            record_latency_breakdown(
                final_info[
                    "latency_breakdown"
                ]
            )

            # =================================================
            # 10. SERIALIZABLE RESPONSE
            # =================================================

            response_data = {
                "answer": answer,
                "refused": refused,
                "citations": [
                    citation.model_dump()
                    for citation in citations
                ],
            }

            # =================================================
            # 11. EXACT CACHE
            # =================================================

            cache.put(
                make_response_cache_key(
                    req.question,
                    req.top_k,
                    req.mode,
                ),
                "lab7.ask",
                {
                    "question":
                        normalize_question(
                            req.question
                        ),
                    "top_k": req.top_k,
                    "mode": req.mode,
                },
                response_data,
            )

            # =================================================
            # 12. SEMANTIC CACHE
            # =================================================

            try:

                # Reuse the same embedding that was
                # already used for semantic lookup/retrieval.
                if query_embedding is None:

                    query_embedding = embed(
                        normalize_question(
                            req.question
                        ),
                        input_type="query",
                    )

                add_semantic_cache_entry(
                    question=req.question,
                    top_k=req.top_k,
                    mode=req.mode,
                    response_data=response_data,
                    question_embedding=query_embedding,
                )

            except Exception:

                # Semantic caching must never
                # break the actual answer.
                pass

            # =================================================
            # 13. TOTAL LATENCY
            # =================================================

            total_latency_ms = (
                time.perf_counter()
                - request_start
            ) * 1000

            STREAM_TOTAL_LATENCIES_MS.append(
                total_latency_ms
            )

            # =================================================
            # 14. FINAL DONE EVENT
            # =================================================

            yield sse_event({
                "type": "done",
                "answer": answer,
                "refused": refused,
                "citations": [
                    citation.model_dump()
                    for citation in citations
                ],
                "validation": validation,
                "latency_ms": round(
                    total_latency_ms,
                    2,
                ),
                "cost_usd": round(
                    cost_usd,
                    8,
                ),
            })

            # IMPORTANT:
            # There is intentionally NO second
            # meta/token/done block here.

        except BudgetExceeded:

            yield sse_event({
                "type": "error",
                "status_code": 429,
                "message":
                    "Request budget exhausted.",
            })

        except Exception as exc:

            yield sse_event({
                "type": "error",
                "status_code": 503,
                "message": str(exc),
            })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":
                "no-cache",
            "Connection":
                "keep-alive",
            "X-Accel-Buffering":
                "no",
        },
    )



# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/health"
)
def health():

    return {
        "status": "ok",

        "uptime_s": round(
            time.time()
            - _STARTED,
            2,
        ),

        "cache":
            cache.stats(),

        "semantic_cache_entries":
            len(
                SEMANTIC_CACHE
            ),

        "semantic_threshold":
            SEMANTIC_THRESHOLD,
    }


# ============================================================
# METRICS
# ============================================================

@app.get(
    "/metrics"
)
def metrics():

    budget = (
        global_budget()
        .as_dict()
    )

    total_requests = (
        CACHE_STATS[
            "exact_hits"
        ]
        + CACHE_STATS[
            "semantic_hits"
        ]
        + CACHE_STATS[
            "misses"
        ]
    )

    cache_hits = (
        CACHE_STATS[
            "exact_hits"
        ]
        + CACHE_STATS[
            "semantic_hits"
        ]
    )

    if total_requests:

        cache_hit_rate = (
            cache_hits
            / total_requests
        )

    else:

        cache_hit_rate = 0.0

    return {

        # ----------------------------------------------------
        # COST
        # ----------------------------------------------------

        "cost_today_usd":
            budget[
                "cost_usd"
            ],

        "cost_per_query_usd":
            (
                budget[
                    "cost_usd"
                ]
                / max(
                    budget[
                        "calls"
                    ],
                    1,
                )
            ),

        # ----------------------------------------------------
        # CACHE
        # ----------------------------------------------------

        "cache": {

            "exact_hits":
                CACHE_STATS[
                    "exact_hits"
                ],

            "semantic_hits":
                CACHE_STATS[
                    "semantic_hits"
                ],

            "misses":
                CACHE_STATS[
                    "misses"
                ],

            "total_requests":
                total_requests,

            "hit_rate":
                cache_hit_rate,

            "semantic_threshold":
                SEMANTIC_THRESHOLD,

            "semantic_entries":
                len(
                    SEMANTIC_CACHE
                ),
        },

        "error_rates": {
            "422_validation": ERROR_STATS["422_validation"],
            "429_budget": ERROR_STATS["429_budget"],
            "503_provider": ERROR_STATS["503_provider"],
            "500_internal": ERROR_STATS["500_internal"],
        },
        "tool_calls": {
            "total": TOOL_STATS["total"],
            "allowed": TOOL_STATS["allowed"],
            "denied": TOOL_STATS["denied"],
        },

        # ----------------------------------------------------
        # SERVICE LATENCY
        # ----------------------------------------------------

        "latency": {

            "p50_ms":
                percentile(
                    REQUEST_LATENCIES_MS,
                    50,
                ),

            "p95_ms":
                percentile(
                    REQUEST_LATENCIES_MS,
                    95,
                ),

            "p99_ms":
                percentile(
                    REQUEST_LATENCIES_MS,
                    99,
                ),

            "samples":
                len(
                    REQUEST_LATENCIES_MS
                ),
        },

        "latency_breakdown": {
            "retrieval_ms": percentile(
                LATENCY_BREAKDOWN["retrieval_ms"], 50
            ),
            "reranking_ms": percentile(
                LATENCY_BREAKDOWN["reranking_ms"], 50
            ),
            "generation_ms": percentile(
                LATENCY_BREAKDOWN["generation_ms"], 50
            ),
            "validation_ms": percentile(
                LATENCY_BREAKDOWN["validation_ms"], 50
            ),
            "repair_generation_ms": percentile(
                LATENCY_BREAKDOWN["repair_generation_ms"], 50
            ),
            "total_ms": percentile(
                LATENCY_BREAKDOWN["total_ms"], 50
            ),
            "samples": len(LATENCY_BREAKDOWN["total_ms"]),
        },

        # ----------------------------------------------------
        # STREAMING LATENCY
        # ----------------------------------------------------

        "streaming": {

            "ttft_p50_ms":
                percentile(
                    STREAM_TTFT_MS,
                    50,
                ),

            "ttft_p95_ms":
                percentile(
                    STREAM_TTFT_MS,
                    95,
                ),

            "total_p50_ms":
                percentile(
                    STREAM_TOTAL_LATENCIES_MS,
                    50,
                ),

            "total_p95_ms":
                percentile(
                    STREAM_TOTAL_LATENCIES_MS,
                    95,
                ),

            "samples":
                len(
                    STREAM_TTFT_MS
                ),
        },

        # ----------------------------------------------------
        # RAG LATENCY BREAKDOWN
        # ----------------------------------------------------

        "latency_breakdown": {

            "retrieval_ms":
                percentile(
                    LATENCY_BREAKDOWN[
                        "retrieval_ms"
                    ],
                    50,
                ),

            "reranking_ms":
                percentile(
                    LATENCY_BREAKDOWN[
                        "reranking_ms"
                    ],
                    50,
                ),

            "generation_ms":
                percentile(
                    LATENCY_BREAKDOWN[
                        "generation_ms"
                    ],
                    50,
                ),

            "validation_ms":
                percentile(
                    LATENCY_BREAKDOWN[
                        "validation_ms"
                    ],
                    50,
                ),

            "repair_generation_ms":
                percentile(
                    LATENCY_BREAKDOWN[
                        "repair_generation_ms"
                    ],
                    50,
                ),

            "total_ms":
                percentile(
                    LATENCY_BREAKDOWN[
                        "total_ms"
                    ],
                    50,
                ),

            "samples":
                len(
                    LATENCY_BREAKDOWN[
                        "total_ms"
                    ]
                ),
        },

        # ----------------------------------------------------
        # MODEL INFORMATION
        # ----------------------------------------------------

        "model_calls":
            budget[
                "calls"
            ],

        "model_cached_calls":
            budget[
                "cached_calls"
            ],

        "unpriced_calls":
            budget[
                "unpriced_calls"
            ],

        "model_latency_p50_ms":
            budget[
                "latency_p50_ms"
            ],

        "model_latency_p95_ms":
            budget[
                "latency_p95_ms"
            ],

        # ----------------------------------------------------
        # SERVICE
        # ----------------------------------------------------

        "uptime_s":
            round(
                time.time()
                - _STARTED,
                2,
            ),
    }


# ============================================================
# STARTUP
# ============================================================

@app.on_event(
    "startup"
)
def startup():

    print("=" * 60)

    print(
        "Aurora Policy Assistant - Lab 7"
    )

    print("=" * 60)

    print(
        "Semantic cache threshold: "
        f"{SEMANTIC_THRESHOLD}"
    )

    # Build retrieval pipeline once.
    pipeline()

    print(
        "Service ready."
    )

    print("=" * 60)