# #!/usr/bin/env python3
# """Lab 5 — the failure classifier.

#     python labs/lab5/diagnose.py --input reports/lab4.json
#     python labs/lab5/diagnose.py --input reports/lab4.json --pareto

# Implements the T4 §5 diagnostic tree. Everything that can be decided by code
# is decided by code; mode 2 needs your eyes and the script says so.
# """
# from __future__ import annotations

# import argparse
# import json
# import sys
# from collections import Counter
# from pathlib import Path

# ROOT = Path(__file__).resolve().parents[2]
# sys.path.insert(0, str(ROOT))

# from labs.lab3.search import load_corpus, load_questions  # noqa: E402
# from labs.lab4.evaluate import build_retriever, judge_correctness
# from labs.lab4.rag import answer_question, answer_with_gold_context

# MODES = {
#     1: "missing_content",
#     2: "chunk_boundary",
#     3: "embedding_mismatch",
#     4: "ranking",
#     5: "reranker",
#     6: "generation",
#     7: "presentation",
# }


# def answer_in_corpus(gold_answer: str, corpus: dict[str, str],
#                      relevant_docs: list[str]) -> bool:
#     """Mode 1 test. Crude keyword overlap, deliberately.

#     TODO: this is a weak test -- it will pass on a paraphrase and fail on a
#     numeric answer expressed differently. Improve it, and say in your report
#     how you know your improvement is better.
#     """
#     text = " ".join(corpus.get(d, "") for d in relevant_docs).lower()
#     if not text:
#         return False
#     tokens = [t for t in gold_answer.lower().split() if len(t) > 4]
#     if not tokens:
#         return True
#     return sum(1 for t in tokens if t.strip(".,;()") in text) / len(tokens) > 0.4


# def classify(row: dict, q: dict, corpus: dict[str, str], *,
#              gold_context_fixes_it: bool | None = None,
#              in_top_30: bool | None = None,
#              dropped_by_reranker: bool | None = None) -> tuple[int, str]:
#     """Walk the T4 §5 diagnostic tree. Returns (mode, evidence).

#     TODO: complete the branches marked TODO. Follow the tree in the handout;
#     do not invent your own ordering, because the ordering is what makes the
#     modes mutually exclusive.
#     """
#     # Mode 7 first: right answer, wrong citation. Check this before anything
#     # else, because a mode-7 failure is not a retrieval failure at all.
#     if row.get("correctness", 0) >= 2 and not row.get("citations_valid", True):
#         return 7, f"correct answer, invalid citations {row.get('invalid_citations')}"

#     # Mode 1: is the answer even in the corpus?
#     if not answer_in_corpus(q["gold_answer"], corpus, q["relevant_docs"]):
#         return 1, "gold answer content not found in the relevant documents"

#     # Mode 6: does gold context fix it?
#     # TODO: if gold_context_fixes_it is False, this is a generation failure.
#     #       Note the direction -- gold context FIXING the answer means
#     #       RETRIEVAL was at fault, not generation. People get this backwards.

#     # TODO Mode 4/5: gold doc in top 30 but not in the final k
#     #       -> 5 if the reranker dropped it, else 4

#     # TODO Mode 3: gold doc not even in the top 30. Confirm by searching for
#     #       the gold chunk's own text -- if THAT retrieves it, the query is the
#     #       problem (mode 3). If it does not, the chunk itself is unfindable
#     #       (mode 2, needs your eyes).

#     return 2, "needs_human_check: open the chunks around the gold answer"


# def pareto(tally: Counter) -> str:
#     total = sum(tally.values()) or 1
#     lines, cum = ["failure mode          n    share   cumulative"], 0
#     for mode, n in tally.most_common():
#         cum += n
#         bar = "█" * round(30 * n / total)
#         lines.append(f"{MODES[mode]:<20} {n:>3}   {n/total:>5.1%}   "
#                      f"{cum/total:>5.1%}  {bar}")
#     return "\n".join(lines)


# def main() -> None:
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--input", default="reports/lab4.json")
#     ap.add_argument("--pareto", action="store_true")
#     ap.add_argument("--save", default="reports/lab5_diagnosis.json")
#     args = ap.parse_args()

#     rows = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
#     questions = {q["id"]: q for q in load_questions(include_unanswerable=True)}
#     corpus = load_corpus()

#     failures = [r for r in rows
#                 if r.get("correctness", 2) < 2 or not r.get("citations_valid", True)]
#     print(f"{len(failures)} failures out of {len(rows)}\n")

#     out, tally = [], Counter()
#     for r in failures:
#         q = questions[r["id"]]
#         mode, evidence = classify(r, q, corpus)
#         tally[mode] += 1
#         out.append({"id": r["id"], "kind": q["kind"], "mode": mode,
#                     "mode_name": MODES[mode], "evidence": evidence,
#                     "question": q["question"], "answer": r["answer"][:300]})
#         print(f"  {r['id']:<5} {MODES[mode]:<20} {evidence}")

#     print("\n" + pareto(tally))
#     print("\nCases marked needs_human_check are Part A2. Open them.")

#     p = ROOT / args.save
#     p.parent.mkdir(parents=True, exist_ok=True)
#     p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
#     print(f"\nsaved -> {p}")


# if __name__ == "__main__":
#     main()







#!/usr/bin/env python3
"""Lab 5 — the failure classifier.

    python labs/lab5/diagnose.py --input reports/lab4.json
    python labs/lab5/diagnose.py --input reports/lab4.json --pareto

Implements the T4 §5 diagnostic tree.

The diagnostic order is:

    Mode 7: presentation / citation failure
        ↓
    Mode 1: answer content missing from corpus
        ↓
    Gold context test
        ├── gold context still fails → Mode 6: generation
        └── gold context fixes it
                ↓
            Gold document in top 30?
                ├── YES → ranking / reranking
                └── NO
                     ↓
                Search using gold text
                    ├── YES → Mode 3: embedding mismatch
                    └── NO  → Mode 2: chunk boundary

Mode 2 is deliberately left as a human-check case because the lab
requires inspection of the chunks around the gold answer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from labs.lab3.search import load_corpus, load_questions
from labs.lab4.evaluate import build_retriever, judge_correctness
from labs.lab4.rag import answer_with_gold_context

MODES = {
    1: "missing_content",
    2: "chunk_boundary",
    3: "embedding_mismatch",
    4: "ranking",
    5: "reranker",
    6: "generation",
    7: "presentation",
}


# ---------------------------------------------------------------------------
# Mode 1
# ---------------------------------------------------------------------------

def normalise_text(text: str) -> str:
    """Normalise text for approximate lexical comparison."""

    text = text.lower()

    # Remove citation markers such as [1], [2], etc.
    text = re.sub(r"\[\d+\]", " ", text)

    # Keep letters, numbers and decimal points.
    text = re.sub(r"[^a-z0-9.%]+", " ", text)

    return " ".join(text.split())


def answer_in_corpus(
    gold_answer: str,
    corpus: dict[str, str],
    relevant_docs: list[str],
) -> bool:
    """Mode 1 test.

    Uses approximate lexical overlap rather than exact string matching.

    The test keeps numeric tokens because insurance answers often depend on
    values such as 36 months, 24 months, 5000, 2%, etc.
    """

    text = " ".join(
        corpus.get(doc_id, "")
        for doc_id in relevant_docs
    )

    text = normalise_text(text)
    gold = normalise_text(gold_answer)

    if not text or not gold:
        return False

    gold_tokens = gold.split()

    # Ignore very common function words.
    stopwords = {
        "the", "and", "that", "this", "with", "from", "under",
        "over", "into", "your", "have", "has", "for", "are",
        "was", "were", "will", "can", "not", "you", "they",
        "their", "there", "than", "then", "also", "only",
        "such", "been", "being", "which", "what", "when",
        "where", "how", "does", "doesnt", "about", "onto",
        "but", "its", "our", "their",
    }

    meaningful = [
        token
        for token in gold_tokens
        if token not in stopwords and len(token) > 2
    ]

    if not meaningful:
        return True

    found = sum(
        1
        for token in meaningful
        if token in text
    )

    overlap = found / len(meaningful)

    # A relatively conservative threshold. The purpose is to avoid
    # declaring a supported answer "missing" simply because it is paraphrased.
    return overlap >= 0.40


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def unique_doc_ids(hits) -> list[str]:
    """Convert chunk hits into a ranked list of unique document IDs."""

    seen = set()
    ranked = []

    for hit in hits:
        if hit.doc_id not in seen:
            seen.add(hit.doc_id)
            ranked.append(hit.doc_id)

    return ranked


def retrieve_top_docs(retriever, question: str, k: int = 30) -> list[str]:
    """Retrieve k chunks and collapse them to unique document IDs."""

    hits = retriever.search(question, k=k)
    return unique_doc_ids(hits)


def gold_doc_in_top_30(
    retriever,
    question: str,
    relevant_docs: list[str],
) -> tuple[bool, list[str]]:
    """Check whether any relevant/gold document appears in top 30."""

    ranked = retrieve_top_docs(
        retriever,
        question,
        k=30,
    )

    relevant = set(relevant_docs)

    found = any(
        doc_id in relevant
        for doc_id in ranked
    )

    return found, ranked


def gold_text_retrieves_itself(
    retriever,
    corpus: dict[str, str],
    relevant_docs: list[str],
) -> tuple[bool, list[str]]:
    """Mode 3 vs Mode 2 test.

    Search using the gold document's own text.

    If the gold document retrieves itself, the query representation is the
    likely problem → Mode 3.

    If the gold document cannot retrieve itself even from its own text,
    leave the case for human inspection → Mode 2.
    """

    all_ranked = []

    for doc_id in relevant_docs:
        gold_text = corpus.get(doc_id, "")

        if not gold_text:
            continue

        ranked = retrieve_top_docs(
            retriever,
            gold_text,
            k=30,
        )

        all_ranked.extend(ranked)

        if doc_id in ranked:
            return True, ranked

    return False, all_ranked


# ---------------------------------------------------------------------------
# Gold-context diagnostic
# ---------------------------------------------------------------------------

def gold_context_fixes(
    q: dict,
    corpus: dict[str, str],
) -> tuple[bool, int, str]:
    """Run the gold-context generation test.

    Returns:
        (fixes, correctness_score, generated_answer)
    """

    gold_docs = [
        corpus[doc_id]
        for doc_id in q["relevant_docs"]
        if doc_id in corpus
    ]

    if not gold_docs:
        return False, 0, ""

    gold_answer = answer_with_gold_context(
        q["question"],
        gold_docs,
    )

    score = judge_correctness(
        q["question"],
        gold_answer.text,
        q["gold_answer"],
    )

    fixes = score >= 2

    return fixes, score, gold_answer.text


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def classify(
    row: dict,
    q: dict,
    corpus: dict[str, str],
    retriever,
    *,
    gold_context_fixes_it: bool | None = None,
    gold_context_score: int | None = None,
    in_top_30: bool | None = None,
    top_30_docs: list[str] | None = None,
    gold_text_finds_itself: bool | None = None,
) -> tuple[int, str]:

    # ------------------------------------------------------------------
    # Mode 7
    # ------------------------------------------------------------------
    # A substantively correct answer with invalid citations is a
    # presentation/citation failure, not a retrieval or generation failure.

    if (
        row.get("correctness", 0) >= 2
        and not row.get("citations_valid", True)
    ):
        return (
            7,
            f"correct answer but invalid citations: "
            f"{row.get('invalid_citations')}",
        )

    # ------------------------------------------------------------------
    # Mode 1
    # ------------------------------------------------------------------

    if not answer_in_corpus(
        q["gold_answer"],
        corpus,
        q["relevant_docs"],
    ):
        return (
            1,
            "gold answer content not found in relevant corpus documents",
        )

    # ------------------------------------------------------------------
    # Mode 6
    # ------------------------------------------------------------------
    # IMPORTANT:
    #
    # gold context FIXING the answer means retrieval was the problem.
    #
    # gold context STILL failing means generation was the problem.

    if gold_context_fixes_it is None:
        return (
            6,
            "gold-context test not yet run",
        )

    if not gold_context_fixes_it:
        return (
            6,
            f"gold context still incorrect "
            f"(gold-context correctness={gold_context_score})",
        )

    # ------------------------------------------------------------------
    # Retrieval problem from here onward
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Mode 4 / 5
    # ------------------------------------------------------------------

    if in_top_30 is None:
        return (
            4,
            "top-30 retrieval test not yet run",
        )

    if in_top_30:
        retrieved_final = set(row.get("retrieved", []))
        relevant = set(q["relevant_docs"])

        # Lab 4 has no reranker in its actual answer pipeline.
        #
        # Therefore, if a relevant document reached retrieval but did not
        # survive the final retrieved set, classify it as ranking.
        #
        # Mode 5 is not assigned here because Lab 4 did not supply a
        # reranker to answer_question().
        if retrieved_final.intersection(relevant):
            return (
                4,
                "gold document retrieved but ranking/final selection "
                "did not produce a sufficient answer",
            )

        return (
            4,
            "gold document is in top 30 but absent from Lab 4 final "
            "retrieved documents; no reranker was used",
        )

    # ------------------------------------------------------------------
    # Mode 3 vs Mode 2
    # ------------------------------------------------------------------

    if gold_text_finds_itself is None:
        return (
            2,
            "needs_human_check: gold document missed top 30; "
            "run gold-text self-retrieval",
        )

    if gold_text_finds_itself:
        return (
            3,
            "gold document misses the question query but retrieves "
            "when searching its own text",
        )

    return (
        2,
        "needs_human_check: gold document cannot retrieve itself; "
        "inspect chunks around the gold answer",
    )


# ---------------------------------------------------------------------------
# Pareto
# ---------------------------------------------------------------------------

def pareto(tally: Counter) -> str:
    total = sum(tally.values()) or 1

    lines = [
        "failure mode          n    share   cumulative"
    ]

    cumulative = 0

    for mode, n in tally.most_common():
        cumulative += n

        bar = "█" * round(
            30 * n / total
        )

        lines.append(
            f"{MODES[mode]:<20} "
            f"{n:>3}   "
            f"{n / total:>5.1%}   "
            f"{cumulative / total:>5.1%}  "
            f"{bar}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        default="reports/lab4.json",
    )

    ap.add_argument(
        "--pareto",
        action="store_true",
    )

    ap.add_argument(
        "--save",
        default="reports/lab5_diagnosis.json",
    )

    args = ap.parse_args()

    # ---------------------------------------------------------------
    # Load Lab 4 results
    # ---------------------------------------------------------------

    rows = json.loads(
        (ROOT / args.input).read_text(
            encoding="utf-8"
        )
    )

    questions = {
        q["id"]: q
        for q in load_questions(
            include_unanswerable=True
        )
    }

    corpus = load_corpus()

    # ---------------------------------------------------------------
    # Build the SAME retriever used by Lab 4
    # ---------------------------------------------------------------

    print("Building Lab 4 retriever...")

    retriever = build_retriever()

    # ---------------------------------------------------------------
    # Identify failures
    # ---------------------------------------------------------------

    failures = [
        r
        for r in rows
        if (
            r.get("correctness", 2) < 2
            or not r.get("citations_valid", True)
        )
    ]

    print(
        f"{len(failures)} failures out of {len(rows)}\n"
    )

    out = []
    tally = Counter()

    # ---------------------------------------------------------------
    # Diagnose every failure
    # ---------------------------------------------------------------

    for index, r in enumerate(failures, start=1):

        q = questions[r["id"]]

        print(
            f"[{index}/{len(failures)}] "
            f"Diagnosing {r['id']}..."
        )

        # -----------------------------------------------------------
        # Gold-context test
        # -----------------------------------------------------------

        fixes, gold_score, gold_answer = gold_context_fixes(
            q,
            corpus,
        )

        # -----------------------------------------------------------
        # Top-30 retrieval test
        # -----------------------------------------------------------

        in_top_30, top_30_docs = gold_doc_in_top_30(
            retriever,
            q["question"],
            q["relevant_docs"],
        )

        # -----------------------------------------------------------
        # Gold-text self-retrieval
        # -----------------------------------------------------------

        if not in_top_30:

            self_retrieved, self_ranked = (
                gold_text_retrieves_itself(
                    retriever,
                    corpus,
                    q["relevant_docs"],
                )
            )

        else:

            self_retrieved = None
            self_ranked = []

        # -----------------------------------------------------------
        # Classify
        # -----------------------------------------------------------

        mode, evidence = classify(
            r,
            q,
            corpus,
            retriever,
            gold_context_fixes_it=fixes,
            gold_context_score=gold_score,
            in_top_30=in_top_30,
            top_30_docs=top_30_docs,
            gold_text_finds_itself=self_retrieved,
        )

        tally[mode] += 1

        result = {
            "id": r["id"],
            "kind": q["kind"],
            "mode": mode,
            "mode_name": MODES[mode],
            "evidence": evidence,

            # Original Lab 4 result
            "lab4_correctness": r.get("correctness"),
            "lab4_citations_valid": r.get(
                "citations_valid"
            ),
            "lab4_retrieved": r.get(
                "retrieved",
                []
            ),
            "relevant_docs": q.get(
                "relevant_docs",
                []
            ),

            # Diagnostic evidence
            "gold_context_correctness": gold_score,
            "gold_context_fixes_it": fixes,
            "in_top_30": in_top_30,
            "top_30_docs": top_30_docs,

            "gold_text_finds_itself": self_retrieved,
            "gold_text_search_ranked_docs": self_ranked,

            # Useful for Part A2
            "needs_human_check": mode == 2,

            "question": q["question"],
            "answer": r["answer"][:500],
            "gold_answer": q["gold_answer"],
        }

        out.append(result)

        print(
            f"  {r['id']:<5} "
            f"{MODES[mode]:<20} "
            f"{evidence}"
        )

    # ---------------------------------------------------------------
    # Print Pareto
    # ---------------------------------------------------------------

    print("\n" + pareto(tally))

    print(
        "\nCases marked needs_human_check are Part A2. "
        "Open the chunks around the gold answer."
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    p = ROOT / args.save

    p.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    p.write_text(
        json.dumps(
            out,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"\nsaved -> {p}"
    )


if __name__ == "__main__":
    main()