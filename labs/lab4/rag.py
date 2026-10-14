#!/usr/bin/env python3
"""Lab 4 — your RAG pipeline.

Write this yourself. `aip/rag.py` is the reference implementation; look at it
after Part A, not before. Labs 5-7 build on whichever of the two you prefer,
but you must be able to explain every line of the one you use.
"""
from __future__ import annotations
import time

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.guards import UNTRUSTED_SYSTEM_CLAUSE, delimit_untrusted  # noqa: E402
from aip.llm import chat  # noqa: E402
from aip.retrieval import Hit, Retriever, format_context  # noqa: E402

# The exact string the system must emit when it cannot answer. Exact, because
# downstream code detects refusal by matching it -- a paraphrase is a bug.
REFUSAL = "I don't have enough information in the provided sources to answer that."

# TODO A: write this before you read aip/rag.py::ANSWER_SYSTEM.
ANSWER_SYSTEM = f"""\
You are a grounded RAG assistant. Answer the user's question using only
the information contained in the supplied sources.

Rules:
1. Use only the supplied sources. Do not use general knowledge, assumptions,
   or information from outside the supplied sources.

2. Identify the policy, plan, product, or category that the question refers to
   before answering. Use only information that is applicable to that scope.

3. Do not combine conditions, limits, waiting periods, benefits, exclusions,
   or other details from different plans or policies unless the question
   explicitly asks for a comparison or the supplied sources require it.

4. Answer all parts of the user's question that are directly supported by
   the supplied sources. Do not stop after answering only one part of a
   multi-part question.

5. Include a detail only when it is supported by the supplied sources and
   relevant to the question. Do not add extra facts merely because they
   appear in the context.

6. When the question asks "which plans", "what plans", or otherwise requires
   a comparison or aggregation, check the supplied evidence for all relevant
   plans before answering. Do not assume that finding one matching plan is
   sufficient.

7. When both current and archived or superseded information is supplied,
   prefer the current/effective information unless the question explicitly
   asks about historical or archived information.

8. Preserve exact numbers, dates, percentages, limits, waiting periods,
   and conditions from the sources. Do not calculate or infer additional
   facts unless the calculation follows directly from the supplied evidence.

9. Cite factual claims using numbered source citations such as [1], [2], [5].

10. A citation number must refer only to a source that was actually supplied.

11. Every factual claim in the answer must have an appropriate citation.

12. If the supplied sources do not contain enough information to answer the
    question, respond exactly:
    {REFUSAL}

13. If the sources support only part of the question, answer the supported
    part and clearly state what information is unavailable. Do not guess or
    invent the missing information.

14. If the sources contain conflicting information, identify the disagreement
    and cite the sources supporting the different claims.

15. Keep the answer to 2–3 sentences unless additional detail is necessary
    to answer the question accurately.

{UNTRUSTED_SYSTEM_CLAUSE}
"""


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit] = field(default_factory=list)
    refused: bool = False
    citations_valid: bool = False
    invalid_citations: list[int] = field(default_factory=list)
    n_citations: int = 0
    truncated: bool = False
    latency_breakdown: dict = field(default_factory=dict)


def validate_answer(
    text: str,
    n_sources: int,
    finish_reason: str | None = None,
) -> dict:
    """Validate citations and generation completeness."""

    citations = [int(n) for n in re.findall(r"\[(\d+)\]", text)]

    invalid_citations = [
        n for n in citations
        if n < 1 or n > n_sources
    ]

    truncated = finish_reason == "length"

    is_refusal = text.strip() == REFUSAL

    missing_citation = not is_refusal and len(citations) == 0

    citations_valid = (
        not invalid_citations
        and not truncated
        and not missing_citation
    )

    return {
        "citations_valid": citations_valid,
        "invalid_citations": invalid_citations,
        "n_citations": len(citations),
        "truncated": truncated,
    }


def answer_question(
    question: str,
    retriever: Retriever,
    *,
    k: int = 10,
    final_k: int = 4,
    reranker=None,
    tier: str = "MAIN",
) -> Answer:
    """Retrieve, generate, validate, and repair/refuse if necessary."""

    total_start = time.perf_counter()

    # ---------------------------------------------------------
    # 1. Retrieve a wider candidate set.
    # ---------------------------------------------------------
    retrieval_start = time.perf_counter()

    hits = retriever.search(question, k=k)

    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

    # ---------------------------------------------------------
    # 2. Optional reranking.
    # ---------------------------------------------------------
    reranking_start = time.perf_counter()

    if reranker is not None:
        hits = reranker.rerank(question, hits)

    reranking_ms = (time.perf_counter() - reranking_start) * 1000

    # ---------------------------------------------------------
    # 3. Keep only the final number of sources.
    # ---------------------------------------------------------
    hits = hits[:final_k]

    # ---------------------------------------------------------
    # 4. Number the retrieved sources so the model can cite them.
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
    # 5. Generate the answer.
    # ---------------------------------------------------------
    generation_start = time.perf_counter()

    response = chat(
        prompt,
        system=ANSWER_SYSTEM,
        tier=tier,
        temperature=0,
        max_tokens=750,
        return_full=True,
    )

    generation_ms = (time.perf_counter() - generation_start) * 1000

    # ---------------------------------------------------------
    # 6. Extract text and finish reason.
    # ---------------------------------------------------------
    if isinstance(response, dict):
        text = response.get("text", "")
        finish_reason = response.get("finish_reason")
    else:
        text = str(response)
        finish_reason = None

    # ---------------------------------------------------------
    # 7. Validate the generated answer.
    # ---------------------------------------------------------
    validation_start = time.perf_counter()

    validation = validate_answer(
        text,
        len(hits),
        finish_reason,
    )

    validation_ms = (time.perf_counter() - validation_start) * 1000

    # ---------------------------------------------------------
    # Create timing information.
    # ---------------------------------------------------------
    latency_breakdown = {
        "retrieval_ms": retrieval_ms,
        "reranking_ms": reranking_ms,
        "generation_ms": generation_ms,
        "validation_ms": validation_ms,
        "repair_generation_ms": 0.0,
    }

    if validation["citations_valid"]:
        latency_breakdown["total_ms"] = (
            (time.perf_counter() - total_start) * 1000
        )

        return Answer(
            question=question,
            text=text,
            hits=hits,
            refused=(text.strip() == REFUSAL),
            citations_valid=True,
            invalid_citations=validation["invalid_citations"],
            n_citations=validation["n_citations"],
            truncated=validation["truncated"],
            latency_breakdown=latency_breakdown,
        )

    # ---------------------------------------------------------
    # 8. Repair once if validation failed.
    # ---------------------------------------------------------
    repair_prompt = f"""\
The previous answer failed citation validation.

Question:
{question}

Sources:
{context}

Previous answer:
{text}

Rewrite the answer using ONLY the supplied sources.

Every factual claim must have a valid citation such as [1] or [2].
Use only citation numbers from 1 to {len(hits)}.

If the sources do not contain enough information to answer the question,
respond exactly:
{REFUSAL}

Do not invent information.
"""

    repair_start = time.perf_counter()

    repaired = chat(
        repair_prompt,
        system=ANSWER_SYSTEM,
        tier=tier,
        temperature=0,
        max_tokens=750,
        return_full=True,
    )

    repair_generation_ms = (
        (time.perf_counter() - repair_start) * 1000
    )

    if isinstance(repaired, dict):
        repaired_text = repaired.get("text", "")
        repaired_finish = repaired.get("finish_reason")
    else:
        repaired_text = str(repaired)
        repaired_finish = None

    # ---------------------------------------------------------
    # Validate repaired answer.
    # ---------------------------------------------------------
    repair_validation_start = time.perf_counter()

    repaired_validation = validate_answer(
        repaired_text,
        len(hits),
        repaired_finish,
    )

    repair_validation_ms = (
        (time.perf_counter() - repair_validation_start) * 1000
    )

    latency_breakdown["repair_generation_ms"] = repair_generation_ms

    # Include repair validation in total validation time.
    latency_breakdown["validation_ms"] += repair_validation_ms

    if repaired_validation["citations_valid"]:
        latency_breakdown["total_ms"] = (
            (time.perf_counter() - total_start) * 1000
        )

        return Answer(
            question=question,
            text=repaired_text,
            hits=hits,
            refused=(repaired_text.strip() == REFUSAL),
            citations_valid=True,
            invalid_citations=repaired_validation["invalid_citations"],
            n_citations=repaired_validation["n_citations"],
            truncated=repaired_validation["truncated"],
            latency_breakdown=latency_breakdown,
        )

    # ---------------------------------------------------------
    # 9. Never return an invalid non-refusal answer.
    # ---------------------------------------------------------
    latency_breakdown["total_ms"] = (
        (time.perf_counter() - total_start) * 1000
    )

    return Answer(
        question=question,
        text=REFUSAL,
        hits=hits,
        refused=True,
        citations_valid=True,
        invalid_citations=[],
        n_citations=0,
        truncated=False,
        latency_breakdown=latency_breakdown,
    )


def answer_with_gold_context(
    question: str,
    gold_docs: list[str],
    *,
    tier: str = "MAIN",
) -> Answer:
    """Generate an answer using gold documents without retrieval."""

    # Gold documents are used directly. No retrieval is performed.
    context = "\n\n".join(
        f"[{i}] {doc}"
        for i, doc in enumerate(gold_docs, start=1)
    )

    prompt = f"""\
Question:
{question}

Sources:
{context}

Answer the question using only the supplied sources.
"""

    response = chat(
        prompt,
        system=ANSWER_SYSTEM,
        tier=tier,
        temperature=0,
        max_tokens=750,
        return_full=True,
    )

    if isinstance(response, dict):
        text = response.get("text", "")
        finish_reason = response.get("finish_reason")
    else:
        text = str(response)
        finish_reason = None

    validation = validate_answer(
        text,
        len(gold_docs),
        finish_reason,
    )

    if validation["citations_valid"]:
        return Answer(
            question=question,
            text=text,
            hits=[],
            refused=(text.strip() == REFUSAL),
            citations_valid=True,
            invalid_citations=validation["invalid_citations"],
            n_citations=validation["n_citations"],
            truncated=validation["truncated"],
        )

    # If the generator produces an invalid answer even with gold context,
    # return the required refusal rather than exposing an invalid citation.
    return Answer(
        question=question,
        text=REFUSAL,
        hits=[],
        refused=True,
        citations_valid=True,
        invalid_citations=[],
        n_citations=0,
        truncated=False,
    )
