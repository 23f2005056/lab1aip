from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from labs.lab7.service import pipeline
from labs.lab4.rag import answer_question


retriever = pipeline()

questions = [
    "Can I get an out-patient physiotherapy session reimbursed on Bronze?",
    "AUR-HI-SIL-2026 — what are the sum insured options?",
]

for q in questions:
    print("=" * 80)
    print("QUESTION:")
    print(q)
    print()

    result = answer_question(
        q,
        retriever,
        k=30,
        final_k=10,
    )

    print("ANSWER:")
    print(result.text)
    print()

    print("REFUSED:", result.refused)
    print("CITATIONS VALID:", result.citations_valid)
    print("N CITATIONS:", result.n_citations)
    print("INVALID CITATIONS:", result.invalid_citations)

    print("\nRETRIEVED:")
    for i, hit in enumerate(result.hits, start=1):
        print(
            i,
            "|",
            getattr(hit, "doc_id", None),
            "|",
            getattr(hit, "score", None),
        )

    print("\nLATENCY:")
    print(result.latency_breakdown)
    print()
