
import json
import requests
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

questions = [
    "What is the reimbursement claim deadline?",
    "When must a reimbursement claim be submitted after discharge?",
    "What documents are required for an emergency reimbursement claim?",
    "What is the process for submitting a reimbursement claim?",
    "How long does reimbursement claim processing take?",
    "What happens when a claim is incomplete?",
    "What is the process for cancelling a policy?",
    "How do I cancel my policy?",
    "What is the deadline for pre-hospitalisation bills?",
    "When should post-hospitalisation bills be submitted?",
    "What expenses are covered before hospitalisation?",
    "What expenses are covered after hospitalisation?",
    "Why might a reimbursement claim be rejected?",
    "How do I submit additional documents for a claim?",
    "What documents are needed for a cashless claim?",
    "What is the cashless claim process?",
    "What happens if I use a non-network hospital?",
    "Which documents are required to file a claim?",
    "What is the process from hospital discharge to payment?",
    "What are the important claim submission deadlines?",
    "How long do I have to respond to an Aurora query?",
    "What is the response deadline for an Aurora query?",
    "How do I escalate an unresolved grievance?",
    "What is the procedure for escalating a grievance?",
    "Which documents are required for KYC?",
    "What documents do I need to complete KYC?",
    "How long does Aurora take to settle a reimbursement claim?",
    "How many days does Aurora need to settle a reimbursement?",
    "What is the deadline to submit a reimbursement claim after discharge?",
    "How long does it take to process a reimbursement claim after discharge?",
]

results = []
for i, question in enumerate(questions, start=1):
    try:
        response = requests.post(
            f"{BASE_URL}/ask",
            json={"question": question, "top_k": 5, "mode": "rag"},
            timeout=120,
        )
        result = {
            "number": i,
            "question": question,
            "status_code": response.status_code,
            "success": response.ok,
        }
        if response.ok:
            data = response.json()
            result["cached"] = data.get("cached")
            result["latency_ms"] = data.get("latency_ms")
        else:
            result["error"] = response.text[:500]

    except Exception as exc:
        result = {
            "number": i,
            "question": question,
            "success": False,
            "error": str(exc),
        }

    results.append(result)
    print(f'{i:02}/30 | success={result["success"]} | {question}')

Path("reports").mkdir(exist_ok=True)
Path("reports/lab7_cache_warmup_30.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\nFinished. Results: reports/lab7_cache_warmup_30.json")
print("Successful requests:", sum(r["success"] for r in results), "/ 30")
