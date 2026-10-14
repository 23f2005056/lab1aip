from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path

import requests


BASE_URL = "http://127.0.0.1:8000"
ENDPOINT = f"{BASE_URL}/ask/stream"

TTFT_TARGET_MS = 1500.0

# QUESTIONS = [
#     "What is the reimbursement claim deadline?",
#     "When must a reimbursement claim be submitted after discharge?",
#     "What documents are required for an emergency reimbursement claim?",
#     "What is the process for submitting a reimbursement claim?",
#     "How long does it take to process a reimbursement claim?",
#     "What happens when a reimbursement claim is incomplete?",
#     "What is the policy cancellation process?",
#     "How can I cancel my policy?",
#     "What is the deadline for submitting pre-hospitalisation bills?",
#     "When should post-hospitalisation bills be submitted?",
#     "What expenses are covered during pre-hospitalisation?",
#     "What expenses are covered after hospitalisation?",
#     "What should I do if my reimbursement claim is rejected?",
#     "Can I submit additional documents after filing a reimbursement claim?",
#     "What are the requirements for a cashless claim?",
#     "How does the cashless hospitalisation process work?",
#     "What happens if the hospital is not part of the network?",
#     "What documents are needed when filing a claim?",
#     "Explain the reimbursement claim process from hospital discharge to payment.",
#     "What are the important deadlines I should know when submitting a health insurance claim?",
# ]

QUESTIONS = [
    "What is the reimbursement claim deadline?",
    "When must a reimbursement claim be submitted after discharge?",
    "What documents are required for an emergency reimbursement claim?",
    "What is the process for submitting a reimbursement claim?",
    "How long does it take to process a reimbursement claim?",
    "What happens when a reimbursement claim is incomplete?",
    "What is the policy cancellation process?",
    "How can I cancel my policy?",
    "What is the deadline for submitting pre-hospitalisation bills?",
    "When should post-hospitalisation bills be submitted?",
    "What expenses are covered during pre-hospitalisation?",
    "What expenses are covered after hospitalisation?",
    "What should I do if my reimbursement claim is rejected?",
    "Can I submit additional documents after filing a reimbursement claim?",
    "What are the requirements for a cashless claim?",
    "How does the cashless hospitalisation process work?",
    "What happens if the hospital is not part of the network?",
    "What documents are needed when filing a claim?",
    "Explain the reimbursement claim process from hospital discharge to payment.",
    "What are the important deadlines I should know when submitting a health insurance claim?",
]

OUTPUT_CSV = Path("reports/lab7_ttft_20q.csv")
OUTPUT_JSON = Path("reports/lab7_ttft_20q.json")


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * p / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)

    fraction = position - lower

    return (
        values[lower]
        + (values[upper] - values[lower]) * fraction
    )


def read_sse_response(response):
    """
    Read an SSE response and measure:

    TTFT:
        Time from request start until the first SSE data/event.

    Total:
        Time until the complete SSE response is received.
    """

    first_event_time = None

    for line in response.iter_lines(decode_unicode=True):

        if not line:
            continue

        if first_event_time is None:
            first_event_time = time.perf_counter()

        # Continue reading until the server finishes.

    end_time = time.perf_counter()

    return first_event_time, end_time


def run_question(question: str, index: int) -> dict:

    payload = {
        "question": question,
        "top_k": 5,
        "mode": "rag",
    }

    start = time.perf_counter()

    result = {
        "index": index,
        "question": question,
        "status_code": None,
        "ttft_ms": None,
        "total_ms": None,
        "cached": None,
        "error": None,
    }

    try:

        with requests.post(
            ENDPOINT,
            json=payload,
            stream=True,
            timeout=120,
            headers={
                "Accept": "text/event-stream",
            },
        ) as response:

            result["status_code"] = response.status_code

            first_event_time = None

            for line in response.iter_lines(
                decode_unicode=True
            ):

                if not line:
                    continue

                if first_event_time is None:
                    first_event_time = time.perf_counter()

            end = time.perf_counter()

            if first_event_time is not None:
                result["ttft_ms"] = (
                    first_event_time - start
                ) * 1000

            result["total_ms"] = (
                end - start
            ) * 1000

            # The service may expose cache information
            # through the SSE payload. Try to parse it.
            #
            # If it is not present, cached remains None.

            if response.status_code != 200:
                result["error"] = (
                    f"HTTP {response.status_code}"
                )

    except Exception as exc:

        result["error"] = str(exc)

        result["total_ms"] = (
            time.perf_counter() - start
        ) * 1000

    return result


def summarize(results: list[dict]) -> dict:

    valid = [
        r
        for r in results
        if r["ttft_ms"] is not None
    ]

    ttfts = [
        r["ttft_ms"]
        for r in valid
    ]

    totals = [
        r["total_ms"]
        for r in results
        if r["total_ms"] is not None
    ]

    passed = [
        r
        for r in valid
        if r["ttft_ms"] <= TTFT_TARGET_MS
    ]

    failed = [
        r
        for r in valid
        if r["ttft_ms"] > TTFT_TARGET_MS
    ]

    summary = {
        "total_questions": len(results),
        "successful_ttft_measurements": len(valid),
        "errors": len(results) - len(valid),

        "ttft_target_ms": TTFT_TARGET_MS,

        "ttft_mean_ms": (
            statistics.mean(ttfts)
            if ttfts else None
        ),

        "ttft_p50_ms": percentile(
            ttfts, 50
        ),

        "ttft_p95_ms": percentile(
            ttfts, 95
        ),

        "ttft_p99_ms": percentile(
            ttfts, 99
        ),

        "ttft_min_ms": (
            min(ttfts)
            if ttfts else None
        ),

        "ttft_max_ms": (
            max(ttfts)
            if ttfts else None
        ),

        "ttft_std_ms": (
            statistics.stdev(ttfts)
            if len(ttfts) > 1
            else 0
        ),

        "ttft_pass_count": len(passed),
        "ttft_fail_count": len(failed),

        "ttft_pass_rate": (
            len(passed) / len(valid)
            if valid else 0
        ),

        "total_latency_mean_ms": (
            statistics.mean(totals)
            if totals else None
        ),

        "total_latency_p50_ms": percentile(
            totals, 50
        ),

        "total_latency_p95_ms": percentile(
            totals, 95
        ),

        "overall_ttft_target_pass": (
            percentile(ttfts, 95) <= TTFT_TARGET_MS
            if ttfts else False
        ),
    }

    return summary


def main():

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("LAB 7 — TTFT AUTOMATED BENCHMARK")
    print("=" * 70)

    print(f"Endpoint: {ENDPOINT}")
    print(f"Questions: {len(QUESTIONS)}")
    print(f"TTFT target: {TTFT_TARGET_MS:.0f} ms")
    print()

    # Check service first.

    try:
        health = requests.get(
            f"{BASE_URL}/health",
            timeout=10,
        )

        print(
            f"Health check: HTTP {health.status_code}"
        )

    except Exception as exc:

        print("ERROR: Could not connect to service.")
        print(exc)
        return

    print()

    results = []

    for index, question in enumerate(
        QUESTIONS,
        start=1,
    ):

        print(
            f"[{index:02d}/{len(QUESTIONS)}] "
            f"{question}"
        )

        result = run_question(
            question,
            index,
        )

        results.append(result)

        if result["ttft_ms"] is not None:

            status = (
                "PASS"
                if result["ttft_ms"]
                <= TTFT_TARGET_MS
                else "FAIL"
            )

            print(
                f"    TTFT:   "
                f"{result['ttft_ms']:.2f} ms "
                f"[{status}]"
            )

        else:

            print(
                f"    TTFT:   ERROR"
            )

        if result["total_ms"] is not None:

            print(
                f"    Total:  "
                f"{result['total_ms']:.2f} ms"
            )

        if result["error"]:

            print(
                f"    Error:  "
                f"{result['error']}"
            )

        print()

    summary = summarize(results)

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "question",
                "status_code",
                "ttft_ms",
                "total_ms",
                "cached",
                "error",
            ],
        )

        writer.writeheader()

        writer.writerows(results)

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------

    output = {
        "summary": summary,
        "results": results,
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("TTFT SUMMARY")
    print("=" * 70)

    for key, value in summary.items():

        if isinstance(value, float):

            print(
                f"{key:<35} "
                f"{value:.3f}"
            )

        else:

            print(
                f"{key:<35} "
                f"{value}"
            )

    print()
    print("=" * 70)

    if summary["overall_ttft_target_pass"]:

        print(
            "RESULT: PASS — "
            "P95 TTFT is within the 1500 ms target."
        )

    else:

        print(
            "RESULT: FAIL — "
            "P95 TTFT exceeds the 1500 ms target."
        )

    print("=" * 70)

    print()
    print(f"CSV:  {OUTPUT_CSV}")
    print(f"JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
    