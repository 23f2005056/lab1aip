#!/usr/bin/env python3
"""Lab 7 — regression gate.

Run:
    AIP_OFFLINE=1 python labs/lab7/gate.py --config labs/lab7/thresholds.yml

The gate combines:
- Lab 4 committed evaluation results
- Lab 3 retrieval evaluation
- Lab 7 recorded service metrics

It exits non-zero when any threshold is breached.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

LAB4_REPORT = ROOT / "reports/lab4.json"
LAB3_REPORT = ROOT / "reports/lab3_sweeps.json"

# Possible Lab 7 metrics files. The first existing one is used.
LAB7_METRICS_CANDIDATES = [
    ROOT / "reports/lab7_metrics.json",
    ROOT / "reports/lab7_metrics_latest.json",
    ROOT / "reports/metrics.json",
]


# ---------------------------------------------------------------------------
# Lab 4 metrics
# ---------------------------------------------------------------------------

def load_lab4_metrics() -> dict[str, float]:
    """Calculate deterministic metrics from the committed Lab 4 report."""

    if not LAB4_REPORT.exists():
        raise FileNotFoundError(
            f"Missing Lab 4 report: {LAB4_REPORT}"
        )

    rows = json.loads(LAB4_REPORT.read_text(encoding="utf-8"))

    if not rows:
        raise ValueError("reports/lab4.json is empty")

    answerable = [
        r for r in rows
        if not r.get("unanswerable", False)
    ]

    unanswerable = [
        r for r in rows
        if r.get("unanswerable", False)
    ]

    refusals = [
        r for r in rows
        if r.get("refused", False)
    ]

    # Correctness is stored as 0, 1, 2.
    correctness = (
        statistics.fmean(r["correctness"] for r in answerable) / 2
        if answerable
        else 0.0
    )

    faithfulness = statistics.fmean(
        r["faithfulness"] for r in rows
    )

    citation_validity = statistics.fmean(
        float(r["citations_valid"])
        for r in rows
    )

    refusal_recall = (
        sum(1 for r in unanswerable if r.get("refused", False))
        / len(unanswerable)
        if unanswerable
        else 0.0
    )

    refusal_precision = (
        sum(1 for r in refusals if r.get("unanswerable", False))
        / len(refusals)
        if refusals
        else 1.0
    )

    return {
        "correctness": correctness,
        "faithfulness": faithfulness,
        "citation_validity": citation_validity,
        "refusal_recall": refusal_recall,
        "refusal_precision": refusal_precision,
    }


# ---------------------------------------------------------------------------
# Lab 3 retrieval metric
# ---------------------------------------------------------------------------

def load_lab3_hit_rate() -> float:
    """Return Hit@5 for the selected Lab 3 markdown-400 dense retriever."""

    if not LAB3_REPORT.exists():
        raise FileNotFoundError(
            f"Missing Lab 3 report: {LAB3_REPORT}"
        )

    data = json.loads(
        LAB3_REPORT.read_text(encoding="utf-8")
    )

    for row in data.get("A2_chunk_size", []):
        if row.get("config") == "markdown-400 dense":
            return float(row["hit_rate@5"])

    raise ValueError(
        "Could not find 'markdown-400 dense' in "
        "A2_chunk_size of reports/lab3_sweeps.json"
    )


# ---------------------------------------------------------------------------
# Lab 7 service metrics
# ---------------------------------------------------------------------------

def load_lab7_metrics() -> dict[str, float]:
    """Load recorded Lab 7 service metrics.

    Cost and latency are deliberately read from a committed metrics artifact
    rather than being invented or inferred from Lab 4 evaluation results.
    """

    metrics_path = next(
        (p for p in LAB7_METRICS_CANDIDATES if p.exists()),
        None,
    )

    if metrics_path is None:
        raise FileNotFoundError(
            "No Lab 7 metrics file found. Expected one of:\n"
            + "\n".join(str(p) for p in LAB7_METRICS_CANDIDATES)
        )

    data = json.loads(
        metrics_path.read_text(encoding="utf-8")
    )

    # Current service /metrics structure.
    cost_per_query = data.get("cost_per_query_usd")


    latency = data.get("latency", {})

    p95_latency = data.get("p95_latency_ms")

    if p95_latency is None:
        p95_latency = latency.get("p95_ms")

    if cost_per_query is None:
        raise ValueError(
            f"Could not find cost_per_query_usd in {metrics_path}"
        )

    if p95_latency is None:
        raise ValueError(
            f"Could not find p95 latency in {metrics_path}"
        )

    return {
        "cost_per_query_usd": float(cost_per_query),
        "p95_latency_ms": float(p95_latency),
    }


# ---------------------------------------------------------------------------
# Main measurement function
# ---------------------------------------------------------------------------

def measure() -> dict[str, float]:
    """Return all metrics required by thresholds.yml."""

    lab4 = load_lab4_metrics()

    try:
        hit_rate_at_5 = load_lab3_hit_rate()
    except (FileNotFoundError, ValueError) as exc:
        print(f"WARNING: {exc}")
        hit_rate_at_5 = float("nan")

    try:
        lab7 = load_lab7_metrics()
    except (FileNotFoundError, ValueError) as exc:
        print(f"WARNING: {exc}")
        lab7 = {
            "cost_per_query_usd": float("nan"),
            "p95_latency_ms": float("nan"),
        }

    return {
        **lab4,
        "hit_rate_at_5": hit_rate_at_5,
        **lab7,
    }


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="labs/lab7/thresholds.yml",
    )
    args = ap.parse_args()

    config_path = ROOT / args.config

    if not config_path.exists():
        print(f"ERROR: threshold file not found: {config_path}")
        return 1

    thresholds = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )

    metrics = measure()

    failures = []

    width = max(len(k) for k in thresholds)

    print(
        f"{'metric':<{width}}  "
        f"{'value':>10}  "
        f"{'gate':>14}  "
        f"status"
    )

    print("-" * (width + 40))

    for name, rule in thresholds.items():

        value = metrics.get(name)

        if value is None:
            failures.append(f"{name}: not measured")

            print(
                f"{name:<{width}}  "
                f"{'—':>10}  "
                f"{'':>14}  "
                f"MISSING"
            )
            continue

        # NaN must fail the gate rather than accidentally passing.
        if isinstance(value, float) and value != value:
            failures.append(
                f"{name}: metric is unavailable"
            )

            print(
                f"{name:<{width}}  "
                f"{'NaN':>10}  "
                f"{'':>14}  "
                f"FAIL"
            )
            continue

        ok = True
        gate = ""

        if "min" in rule:
            gate = f">= {rule['min']}"
            ok = value >= rule["min"]

        if "max" in rule and ok:
            gate = f"<= {rule['max']}"
            ok = value <= rule["max"]

        if not ok:
            failures.append(
                f"{name}: {value} violates {gate}"
            )

        print(
            f"{name:<{width}}  "
            f"{value:>10.4f}  "
            f"{gate:>14}  "
            f"{'ok' if ok else 'FAIL'}"
        )

    if failures:
        print("\nGATE FAILED:")

        for failure in failures:
            print("  " + failure)

        return 1

    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())