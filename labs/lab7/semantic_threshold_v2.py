import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import requests


BASE_URL = "http://127.0.0.1:8000"

THRESHOLDS = [

    0.95

]

RESULTS_DIR = Path("reports")
JSON_PATH = RESULTS_DIR / "semantic_threshold_v2.json"
CSV_PATH = RESULTS_DIR / "semantic_threshold_v2.csv"


# ============================================================
# TEST CASES
# ============================================================

CASES = [

    # --------------------------------------------------------
    # INTENT 1: Reimbursement submission deadline
    # --------------------------------------------------------

    {
        "id": "R01-SEED",
        "category": "seed",
        "intent": "reimbursement_submission_deadline",
        "question": "What is the deadline for submitting a reimbursement claim?",
    },
    {
        "id": "R01-P01",
        "category": "positive",
        "intent": "reimbursement_submission_deadline",
        "question": "How many days do I have to file a reimbursement claim?",
    },
    {
        "id": "R01-P02",
        "category": "positive",
        "intent": "reimbursement_submission_deadline",
        "question": "How long after discharge can I submit the reimbursement claim?",
    },
    {
        "id": "R01-P03",
        "category": "positive",
        "intent": "reimbursement_submission_deadline",
        "question": "When must a reimbursement claim be submitted?",
    },
    {
        "id": "R01-P04",
        "category": "positive",
        "intent": "reimbursement_submission_deadline",
        "question": "What is the allowed period for filing a reimbursement claim?",
    },
    {
        "id": "R01-N01",
        "category": "near_negative",
        "intent": "reimbursement_submission_deadline",
        "question": "How long does Aurora take to settle a reimbursement claim?",
    },
    {
        "id": "R01-N02",
        "category": "near_negative",
        "intent": "reimbursement_submission_deadline",
        "question": "What documents are required for a reimbursement claim?",
    },


    # --------------------------------------------------------
    # INTENT 2: Aurora query response deadline
    # --------------------------------------------------------

    {
        "id": "R02-SEED",
        "category": "seed",
        "intent": "aurora_query_response",
        "question": "What is the deadline for responding to an Aurora query?",
    },
    {
        "id": "R02-P01",
        "category": "positive",
        "intent": "aurora_query_response",
        "question": "How many days do I have to respond to an Aurora query?",
    },
    {
        "id": "R02-P02",
        "category": "positive",
        "intent": "aurora_query_response",
        "question": "How long can I take to answer an Aurora query?",
    },
    {
        "id": "R02-P03",
        "category": "positive",
        "intent": "aurora_query_response",
        "question": "When must I provide a response to an Aurora query?",
    },
    {
        "id": "R02-P04",
        "category": "positive",
        "intent": "aurora_query_response",
        "question": "What is the permitted response period for an Aurora query?",
    },
    {
        "id": "R02-N01",
        "category": "near_negative",
        "intent": "aurora_query_response",
        "question": "What happens after Aurora raises a query?",
    },
    {
        "id": "R02-N02",
        "category": "near_negative",
        "intent": "aurora_query_response",
        "question": "How long does Aurora take to resolve a query?",
    },


    # --------------------------------------------------------
    # INTENT 3: Post-hospitalisation bills
    # --------------------------------------------------------

    {
        "id": "R03-SEED",
        "category": "seed",
        "intent": "post_hospitalisation_submission",
        "question": "When must post-hospitalisation bills be submitted?",
    },
    {
        "id": "R03-P01",
        "category": "positive",
        "intent": "post_hospitalisation_submission",
        "question": "How many days do I have to submit post-hospitalisation bills?",
    },
    {
        "id": "R03-P02",
        "category": "positive",
        "intent": "post_hospitalisation_submission",
        "question": "What is the deadline for post-hospitalisation bills?",
    },
    {
        "id": "R03-P03",
        "category": "positive",
        "intent": "post_hospitalisation_submission",
        "question": "How long after the post-hospitalisation period can I submit the bills?",
    },
    {
        "id": "R03-P04",
        "category": "positive",
        "intent": "post_hospitalisation_submission",
        "question": "When should post-hospitalisation expenses be filed?",
    },
    {
        "id": "R03-N01",
        "category": "near_negative",
        "intent": "post_hospitalisation_submission",
        "question": "What expenses are covered during post-hospitalisation?",
    },
    {
        "id": "R03-N02",
        "category": "near_negative",
        "intent": "post_hospitalisation_submission",
        "question": "How long is the post-hospitalisation coverage period?",
    },


    # --------------------------------------------------------
    # INTENT 4: Grievance escalation
    # --------------------------------------------------------

    {
        "id": "R04-SEED",
        "category": "seed",
        "intent": "grievance_escalation",
        "question": "What is the grievance escalation procedure?",
    },
    {
        "id": "R04-P01",
        "category": "positive",
        "intent": "grievance_escalation",
        "question": "How do I escalate an unresolved grievance?",
    },
    {
        "id": "R04-P02",
        "category": "positive",
        "intent": "grievance_escalation",
        "question": "What should I do if my grievance remains unresolved?",
    },
    {
        "id": "R04-P03",
        "category": "positive",
        "intent": "grievance_escalation",
        "question": "How can an unresolved customer complaint be escalated?",
    },
    {
        "id": "R04-P04",
        "category": "positive",
        "intent": "grievance_escalation",
        "question": "Where should I take a grievance that has not been resolved?",
    },
    {
        "id": "R04-N01",
        "category": "near_negative",
        "intent": "grievance_escalation",
        "question": "Who handles an escalated grievance?",
    },
    {
        "id": "R04-N02",
        "category": "near_negative",
        "intent": "grievance_escalation",
        "question": "How long does grievance resolution take?",
    },


    # --------------------------------------------------------
    # INTENT 5: KYC
    # --------------------------------------------------------

    {
        "id": "R05-SEED",
        "category": "seed",
        "intent": "kyc_requirement",
        "question": "What documents are required for KYC?",
    },
    {
        "id": "R05-P01",
        "category": "positive",
        "intent": "kyc_requirement",
        "question": "Which documents do I need to complete KYC?",
    },
    {
        "id": "R05-P02",
        "category": "positive",
        "intent": "kyc_requirement",
        "question": "What paperwork is needed for KYC verification?",
    },
    {
        "id": "R05-P03",
        "category": "positive",
        "intent": "kyc_requirement",
        "question": "What identification documents are needed for KYC?",
    },
    {
        "id": "R05-P04",
        "category": "positive",
        "intent": "kyc_requirement",
        "question": "What do I need to provide for KYC?",
    },
    {
        "id": "R05-N01",
        "category": "near_negative",
        "intent": "kyc_requirement",
        "question": "How long does KYC verification take?",
    },
    {
        "id": "R05-N02",
        "category": "near_negative",
        "intent": "kyc_requirement",
        "question": "Why is KYC verification required?",
    },


    # --------------------------------------------------------
    # INTENT 6: Claim settlement
    # --------------------------------------------------------

    {
        "id": "R06-SEED",
        "category": "seed",
        "intent": "claim_settlement",
        "question": "How long does Aurora take to settle a complete reimbursement claim?",
    },
    {
        "id": "R06-P01",
        "category": "positive",
        "intent": "claim_settlement",
        "question": "What is the settlement time for a complete reimbursement claim?",
    },
    {
        "id": "R06-P02",
        "category": "positive",
        "intent": "claim_settlement",
        "question": "How many days does Aurora have to settle a reimbursement claim?",
    },
    {
        "id": "R06-P03",
        "category": "positive",
        "intent": "claim_settlement",
        "question": "When should a complete reimbursement claim be settled?",
    },
    {
        "id": "R06-P04",
        "category": "positive",
        "intent": "claim_settlement",
        "question": "What is Aurora's timeline for settling a complete claim?",
    },
    {
        "id": "R06-N01",
        "category": "near_negative",
        "intent": "claim_settlement",
        "question": "What is the deadline for submitting a reimbursement claim?",
    },
    {
        "id": "R06-N02",
        "category": "near_negative",
        "intent": "claim_settlement",
        "question": "What documents are needed before a claim can be settled?",
    },


    # --------------------------------------------------------
    # UNRELATED
    # --------------------------------------------------------

    {
        "id": "U01",
        "category": "unrelated",
        "intent": "unrelated",
        "question": "What is the process for updating my bank account details?",
    },
    {
        "id": "U02",
        "category": "unrelated",
        "intent": "unrelated",
        "question": "How can I change my registered mobile number?",
    },
    {
        "id": "U03",
        "category": "unrelated",
        "intent": "unrelated",
        "question": "What are the rules for cashless hospital admission?",
    },
    {
        "id": "U04",
        "category": "unrelated",
        "intent": "unrelated",
        "question": "How can I download my policy document?",
    },
    {
        "id": "U05",
        "category": "unrelated",
        "intent": "unrelated",
        "question": "What is the procedure for cancelling my policy?",
    },
    {
        "id": "U06",
        "category": "unrelated",
        "intent": "unrelated",
        "question": "How do I update my nominee information?",
    },
]


# ============================================================
# HELPERS
# ============================================================

def ask(question: str) -> dict:
    response = requests.post(
        f"{BASE_URL}/ask",
        json={
            "question": question,
            "top_k": 5,
            "mode": "rag",
        },
        timeout=120,
    )

    response.raise_for_status()
    return response.json()


def health() -> dict:
    response = requests.get(
        f"{BASE_URL}/health",
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def clear_semantic_cache() -> None:
    """
    Ask the running service to clear only its in-memory semantic cache.

    This requires the service to expose a development/test endpoint.
    If it does not, the experiment will stop instead of silently
    contaminating the measurements.
    """

    response = requests.post(
        f"{BASE_URL}/lab7/semantic-cache/clear",
        timeout=10,
    )

    if response.status_code != 200:
        raise RuntimeError(
            "Semantic-cache clear endpoint is not available. "
            "Do not run the experiment with a contaminated cache."
        )


def write_results(results: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    fieldnames = [
        "threshold",
        "case_id",
        "category",
        "intent",
        "question",
        "cached",
        "semantic_similarity",
        "semantic_match",
        "latency_ms",
        "cost_usd",
        "answer",
    ]

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in results:
            writer.writerow(
                {
                    key: row.get(key)
                    for key in fieldnames
                }
            )


def print_summary(results: list[dict]) -> None:

    print()
    print("=" * 80)
    print("SEMANTIC CACHE THRESHOLD EXPERIMENT V2")
    print("=" * 80)

    for threshold in THRESHOLDS:

        rows = [
            r
            for r in results
            if r["threshold"] == threshold
        ]

        positive = [
            r for r in rows
            if r["category"] == "positive"
        ]

        near_negative = [
            r for r in rows
            if r["category"] == "near_negative"
        ]

        unrelated = [
            r for r in rows
            if r["category"] == "unrelated"
        ]

        positive_hits = sum(
            1
            for r in positive
            if r["cached"]
            and r["semantic_similarity"] is not None
        )

        near_negative_hits = sum(
            1
            for r in near_negative
            if r["cached"]
            and r["semantic_similarity"] is not None
        )

        unrelated_hits = sum(
            1
            for r in unrelated
            if r["cached"]
            and r["semantic_similarity"] is not None
        )

        positive_rate = (
            positive_hits / len(positive)
            if positive
            else 0
        )

        near_negative_rate = (
            near_negative_hits / len(near_negative)
            if near_negative
            else 0
        )

        unrelated_rate = (
            unrelated_hits / len(unrelated)
            if unrelated
            else 0
        )

        positive_sims = [
            r["semantic_similarity"]
            for r in positive
            if r["semantic_similarity"] is not None
        ]

        near_negative_sims = [
            r["semantic_similarity"]
            for r in near_negative
            if r["semantic_similarity"] is not None
        ]

        print()
        print(f"Threshold: {threshold:.2f}")
        print("-" * 80)

        print(
            f"Positive semantic hit rate: "
            f"{positive_hits}/{len(positive)} "
            f"({positive_rate:.2%})"
        )

        print(
            f"Near-negative semantic hit rate: "
            f"{near_negative_hits}/{len(near_negative)} "
            f"({near_negative_rate:.2%})"
        )

        print(
            f"Unrelated semantic hit rate: "
            f"{unrelated_hits}/{len(unrelated)} "
            f"({unrelated_rate:.2%})"
        )

        if positive_sims:
            print(
                "Positive similarity range: "
                f"{min(positive_sims):.4f} - "
                f"{max(positive_sims):.4f}"
            )

        if near_negative_sims:
            print(
                "Near-negative similarity range: "
                f"{min(near_negative_sims):.4f} - "
                f"{max(near_negative_sims):.4f}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("Checking Lab 7 service...")

    h = health()

    print(
        f"Service status: {h.get('status')}"
    )

    print(
        f"Current semantic threshold: "
        f"{h.get('semantic_threshold')}"
    )

    print()

    results = []

    for threshold in THRESHOLDS:

        print("=" * 80)
        print(
            f"RUNNING THRESHOLD = {threshold:.2f}"
        )
        print("=" * 80)

        print(
            "NOTE: The running service must be restarted "
            "with SEMANTIC_THRESHOLD set to this value."
        )

        input(
            "Set the threshold, restart the service, "
            "then press ENTER to continue..."
        )

        h = health()

        actual_threshold = h.get(
            "semantic_threshold"
        )

        if actual_threshold != threshold:

            raise RuntimeError(
                f"Expected threshold {threshold}, "
                f"but service reports {actual_threshold}. "
                f"Stop the experiment."
            )

        clear_semantic_cache()

        print(
            "Semantic cache cleared."
        )

        # ----------------------------------------------------
        # Create one seed per intent.
        # ----------------------------------------------------

        seeds = [
            c
            for c in CASES
            if c["category"] == "seed"
        ]

        print(
            f"Creating {len(seeds)} fresh seed entries..."
        )

        for case in seeds:

            print(
                f"  SEED {case['id']}: "
                f"{case['question']}"
            )

            result = ask(
                case["question"]
            )

            # The seed should normally be a cache miss.
            # We don't count it in the summary.
            time.sleep(0.1)

        print()

        # ----------------------------------------------------
        # Test all non-seed cases.
        # ----------------------------------------------------

        test_cases = [
            c
            for c in CASES
            if c["category"] != "seed"
        ]

        for index, case in enumerate(
            test_cases,
            start=1,
        ):

            print(
                f"[{index}/{len(test_cases)}] "
                f"{case['id']} "
                f"({case['category']})"
            )

            print(
                f"  Q: {case['question']}"
            )

            result = ask(
                case["question"]
            )

            row = {
                "threshold": threshold,
                "case_id": case["id"],
                "category": case["category"],
                "intent": case["intent"],
                "question": case["question"],
                "cached": result.get(
                    "cached"
                ),
                "semantic_similarity": result.get(
                    "semantic_similarity"
                ),
                "semantic_match": result.get(
                    "semantic_match"
                ),
                "latency_ms": result.get(
                    "latency_ms"
                ),
                "cost_usd": result.get(
                    "cost_usd"
                ),
                "answer": result.get(
                    "answer",
                    "",
                ),
            }

            results.append(row)

            print(
                f"  cached: "
                f"{row['cached']}"
            )

            print(
                f"  similarity: "
                f"{row['semantic_similarity']}"
            )

            print(
                f"  match: "
                f"{row['semantic_match']}"
            )

            time.sleep(0.1)

        write_results(results)

        print()
        print(
            f"Saved intermediate results to:"
        )
        print(
            f"  {JSON_PATH}"
        )
        print(
            f"  {CSV_PATH}"
        )

    print_summary(results)

    print()
    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    print(
        f"JSON: {JSON_PATH}"
    )

    print(
        f"CSV : {CSV_PATH}"
    )


if __name__ == "__main__":
    main()