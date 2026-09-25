#!/usr/bin/env python3
"""Lab 2 — the configurations under test.

Each variant is a callable `str -> dict`. `grid.py` runs them all through the
same harness, so the only thing that differs between rows of your table is the
thing you intended to differ.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from labs.lab1.extract import (
    SYSTEM_PROMPT,
    TicketRecord,
    TicketRecordC,
    apply_business_rules,
    extract_deterministic,
    structured,
)

# ---------------------------------------------------------------------------
# A1 — your six chosen examples.
# ---------------------------------------------------------------------------
# TODO A1: choose 6 dev-set tickets. For EACH, write one line saying what it
#          teaches that prose cannot. Pick edges, not averages (T2 §2.2):
#            - the billing/complaint boundary
#            - a ticket with no policy number (teaches null)
#            - a Hinglish ticket
#            - a satisfied-but-urgent ticket (the sentiment/urgency trap)
#            - a ticket whose policy number is only in a quoted reply
#            - one you got wrong in Lab 1

FEW_SHOT_IDS: list[str] = [
    "T0097",   # billing/complaint boundary
    "T0054",   # no policy number -> null
    "T0200",   # Hinglish
    "T0222",   # satisfied sentiment vs urgency
    "T0238",   # quoted reply must not become current evidence
    "T0214",   # difficult mis-selling complaint
]


def load_examples(ids: list[str]) -> list[dict]:
    rows = [json.loads(l) for l in
            (ROOT / "data/eval/extraction_dev.jsonl").open(encoding="utf-8")]
    by_id = {r["id"]: r for r in rows}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"unknown example ids: {missing}")
    return [by_id[i] for i in ids]


def few_shot_block(ids: list[str]) -> str:
    examples = load_examples(ids)

    blocks = []

    for i, example in enumerate(examples, 1):
        blocks.append(
            f"EXAMPLE {i}\n"
            f"Ticket:\n{example['input']}\n\n"
            f"Output:\n"
            f"{json.dumps(example['expected'], ensure_ascii=False, indent=2)}"
        )

    return (
        "\n\n"
        "Here are examples showing exactly how tickets should be extracted:\n\n"
        + "\n\n".join(blocks)
        + "\n\n"
        "Now extract the current ticket using the same output format and rules."
    )


# ---------------------------------------------------------------------------
# The variants
# ---------------------------------------------------------------------------
def zero_shot(ticket: str, tier: str = "SMALL") -> dict:
    rec = structured(
        ticket,
        schema=TicketRecordC,
        system=SYSTEM_PROMPT,
        tier=tier,
    )

    result = rec.model_dump()

    result.update(extract_deterministic(ticket))
    result = apply_business_rules(result, ticket)

    return result


def few_shot(ticket: str, tier: str = "SMALL") -> dict:
    system = SYSTEM_PROMPT + few_shot_block(FEW_SHOT_IDS)

    rec = structured(
        ticket,
        schema=TicketRecordC,
        system=system,
        tier=tier,
    )

    result = rec.model_dump()

    result.update(extract_deterministic(ticket))
    result = apply_business_rules(result, ticket)

    return result


class TicketRecordReasoned(TicketRecord):
    """TODO B: add a `reasoning: str` field FIRST (T2 §3.3).

    Pydantic keeps declaration order, and field order in the JSON Schema
    influences generation order. Putting reasoning first makes it condition the
    answer; putting it last makes it a post-hoc rationalisation. You want the
    first. Measure the difference in output tokens.
    """
    reasoning: str




def few_shot_reasoned(ticket: str, tier: str = "SMALL") -> dict:
    system = (
        SYSTEM_PROMPT
        + few_shot_block(FEW_SHOT_IDS)
        + "\nProvide a brief reasoning field first, then the final extraction."
    )

    rec = structured(
        ticket,
        schema=TicketRecordReasoned,
        system=system,
        tier=tier,
    )

    result = rec.model_dump()

    result.update(extract_deterministic(ticket))
    result = apply_business_rules(result, ticket)

    return result


def cascade(ticket: str) -> dict:
    small_1 = few_shot(ticket, "SMALL")
    small_2 = few_shot(ticket, "SMALL")   # second stochastic sample

    important_fields = [
        "category",
        "urgency",
        "sentiment",
        "product",
        "language",
        "policy_number",
        "contains_pii",
        "escalate",
    ]

    agree = all(
        small_1.get(f) == small_2.get(f)
        for f in important_fields
    )

    evidence_ok = bool(str(small_1.get("evidence", "")).strip())

    if agree and evidence_ok:
        small_1["_path"] = "small"
        return small_1

    result = few_shot(ticket, "MAIN")
    result["_path"] = "large"
    return result


VARIANTS = {
    "zero_shot": lambda t: zero_shot(t, "SMALL"),
    "zero_shot_main": lambda t: zero_shot(t, "MAIN"),
    "few_shot": lambda t: few_shot(t, "SMALL"),
    "few_shot_main": lambda t: few_shot(t, "MAIN"),
    "few_shot_reasoned": lambda t: few_shot_reasoned(t, "SMALL"),
    "few_shot_reasoned_main": lambda t: few_shot_reasoned(t, "MAIN"),
    "cascade": cascade,
}
