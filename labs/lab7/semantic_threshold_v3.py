import json
import time
import requests

BASE_URL = "http://127.0.0.1:8000"

# These are compared ONLY against the six fixed seed questions
# already stored in the semantic cache.

TEST_CASES = [

    # =========================================================
    # REIMBURSEMENT CLAIM
    # Seed:
    # "What is the deadline for submitting a reimbursement claim?"
    # =========================================================

    {
        "id": "R01",
        "category": "positive",
        "intent": "reimbursement_deadline",
        "question": "How much time do I have to submit a reimbursement claim?",
    },
    {
        "id": "R02",
        "category": "positive",
        "intent": "reimbursement_deadline",
        "question": "How many days are allowed for filing a reimbursement claim?",
    },
    {
        "id": "R03",
        "category": "positive",
        "intent": "reimbursement_deadline",
        "question": "When is a reimbursement claim due?",
    },
    {
        "id": "R04",
        "category": "positive",
        "intent": "reimbursement_deadline",
        "question": "By what date must I submit my reimbursement claim?",
    },

    {
        "id": "R05",
        "category": "near_negative",
        "intent": "reimbursement_deadline",
        "question": "What documents are needed to submit a reimbursement claim?",
    },
    {
        "id": "R06",
        "category": "near_negative",
        "intent": "reimbursement_deadline",
        "question": "Who approves a reimbursement claim?",
    },


    # =========================================================
    # AURORA QUERY RESPONSE
    # =========================================================

    {
        "id": "A01",
        "category": "positive",
        "intent": "aurora_query",
        "question": "How long do I have to respond to an Aurora query?",
    },
    {
        "id": "A02",
        "category": "positive",
        "intent": "aurora_query",
        "question": "What is the response deadline for an Aurora query?",
    },
    {
        "id": "A03",
        "category": "positive",
        "intent": "aurora_query",
        "question": "When should I respond to an Aurora query?",
    },
    {
        "id": "A04",
        "category": "positive",
        "intent": "aurora_query",
        "question": "How many days are available to answer an Aurora query?",
    },

    {
        "id": "A05",
        "category": "near_negative",
        "intent": "aurora_query",
        "question": "Who sends an Aurora query?",
    },
    {
        "id": "A06",
        "category": "near_negative",
        "intent": "aurora_query",
        "question": "What documents are required for an Aurora query?",
    },


    # =========================================================
    # POST-HOSPITALISATION
    # =========================================================

    {
        "id": "H01",
        "category": "positive",
        "intent": "post_hospitalisation",
        "question": "When should post-hospitalisation bills be submitted?",
    },
    {
        "id": "H02",
        "category": "positive",
        "intent": "post_hospitalisation",
        "question": "How long do I have to submit post-hospitalisation bills?",
    },
    {
        "id": "H03",
        "category": "positive",
        "intent": "post_hospitalisation",
        "question": "What is the deadline for post-hospitalisation expenses?",
    },
    {
        "id": "H04",
        "category": "positive",
        "intent": "post_hospitalisation",
        "question": "By when must post-hospitalisation bills be filed?",
    },

    {
        "id": "H05",
        "category": "near_negative",
        "intent": "post_hospitalisation",
        "question": "What documents are required to submit a hospitalization claim?",
    },
    {
        "id": "H06",
        "category": "near_negative",
        "intent": "post_hospitalisation",
        "question": "What documents are needed for post-hospitalisation expenses?",
    },


    # =========================================================
    # GRIEVANCE ESCALATION
    # =========================================================

    {
        "id": "G01",
        "category": "positive",
        "intent": "grievance",
        "question": "How do I escalate an unresolved grievance?",
    },
    {
        "id": "G02",
        "category": "positive",
        "intent": "grievance",
        "question": "What is the procedure for escalating a grievance?",
    },
    {
        "id": "G03",
        "category": "positive",
        "intent": "grievance",
        "question": "What happens when a grievance remains unresolved?",
    },
    {
        "id": "G04",
        "category": "positive",
        "intent": "grievance",
        "question": "Who should I contact when my grievance needs escalation?",
    },

    {
        "id": "G05",
        "category": "near_negative",
        "intent": "grievance",
        "question": "How do I submit a new grievance?",
    },
    {
        "id": "G06",
        "category": "near_negative",
        "intent": "grievance",
        "question": "What information is required to file a grievance?",
    },


    # =========================================================
    # KYC
    # =========================================================

    {
        "id": "K01",
        "category": "positive",
        "intent": "kyc_documents",
        "question": "Which documents are required for KYC?",
    },
    {
        "id": "K02",
        "category": "positive",
        "intent": "kyc_documents",
        "question": "What documents do I need to complete KYC?",
    },
    {
        "id": "K03",
        "category": "positive",
        "intent": "kyc_documents",
        "question": "What paperwork is needed for KYC verification?",
    },
    {
        "id": "K04",
        "category": "positive",
        "intent": "kyc_documents",
        "question": "Which documents should I provide for KYC?",
    },

    {
        "id": "K05",
        "category": "near_negative",
        "intent": "kyc_documents",
        "question": "How long does KYC verification take?",
    },
    {
        "id": "K06",
        "category": "near_negative",
        "intent": "kyc_documents",
        "question": "Who performs the KYC verification?",
    },


    # =========================================================
    # AURORA SETTLEMENT
    # =========================================================

    {
        "id": "S01",
        "category": "positive",
        "intent": "aurora_settlement",
        "question": "How long does Aurora take to settle a reimbursement claim?",
    },
    {
        "id": "S02",
        "category": "positive",
        "intent": "aurora_settlement",
        "question": "How many days does Aurora need to settle a reimbursement?",
    },
    {
        "id": "S03",
        "category": "positive",
        "intent": "aurora_settlement",
        "question": "What is the settlement time for a complete reimbursement claim?",
    },
    {
        "id": "S04",
        "category": "positive",
        "intent": "aurora_settlement",
        "question": "When will a complete reimbursement claim be settled?",
    },

    {
        "id": "S05",
        "category": "near_negative",
        "intent": "aurora_settlement",
        "question": "What documents are required before Aurora can settle a claim?",
    },
    {
        "id": "S06",
        "category": "near_negative",
        "intent": "aurora_settlement",
        "question": "Who handles reimbursement claim settlement?",
    },


    # =========================================================
    # UNRELATED
    # =========================================================

    {
        "id": "U01",
        "category": "unrelated",
        "intent": "unrelated",
        "question": "What is the capital of France?",
    },
    {
        "id": "U02",
        "category": "unrelated",
        "intent": "unrelated",
        "question": "How does photosynthesis work?",
    },
    {
        "id": "U03",
        "category": "unrelated",
        "intent": "unrelated",
        "question": "What is the largest planet in the solar system?",
    },
    {
        "id": "U04",
        "category": "unrelated",
        "intent": "unrelated",
        "question": "How do I bake a chocolate cake?",
    },
    {
        "id": "U05",
        "category": "unrelated",
        "intent": "unrelated",
        "question": "What is the difference between TCP and UDP?",
    },
    {
        "id": "U06",
        "category": "unrelated",
        "intent": "unrelated",
        "question": "Who wrote Pride and Prejudice?",
    },
]


THRESHOLDS = [
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]

def evaluate_case(case, threshold):
    start = time.perf_counter()

    response = requests.post(
        f"{BASE_URL}/lab7/semantic-cache/eval",
        params={"threshold": threshold},
        json={
            "question": case["question"],
            "top_k": 5,
            "mode": "rag",
        },
        timeout=30,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    response.raise_for_status()

    data = response.json()

    if abs(data["threshold_used"] - threshold) > 1e-9:
        raise RuntimeError(
            f"Threshold mismatch: requested={threshold}, "
            f"used={data['threshold_used']}"
        )

    return {
        **case,
        "cached": data["cached"],
        "semantic_similarity": data["semantic_similarity"],
        "semantic_match": data["semantic_match"],
        "best_match_similarity": data["best_match_similarity"],
        "entries_checked": data["entries_checked"],
        "threshold_used": data["threshold_used"],
        "latency_ms": elapsed_ms,
    }





def main():

    print("=" * 70)
    print("LAB 7 - SEMANTIC CACHE THRESHOLD V3")
    print("=" * 70)

    health = requests.get(
        f"{BASE_URL}/health",
        timeout=10,
    ).json()

    print("Service status:", health["status"])
    print("Semantic entries:", health["semantic_cache_entries"])

    if health["semantic_cache_entries"] != 6:
        raise RuntimeError(
            "Expected exactly 6 fixed semantic-cache seed entries."
        )

    all_results = []

    for threshold in THRESHOLDS:

        for case in TEST_CASES:

            result = evaluate_case(case, threshold)

            all_results.append({
                "threshold": threshold,
                **result,
            })

            similarity = result["semantic_similarity"]

            print(
                f'{case["id"]:>3} '
                f'{case["category"]:<13} '
                f'sim={similarity if similarity is not None else None} '
                f'cached={result["cached"]} '
                f'match={result["semantic_match"]}'
            )

    with open(
        "reports/lab7_semantic_threshold_v3.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            all_results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(
        "Results saved to "
        "reports/lab7_semantic_threshold_v3.json"
    )


if __name__ == "__main__":
    main()