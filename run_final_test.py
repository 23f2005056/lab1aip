import json
import subprocess
import time
import statistics

with open("final_questions1.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

results = []

for i, item in enumerate(questions, 1):
    request = {
        "question": item["question"],
        "top_k": 5,
        "mode": "rag",
    }

    with open("_current_request.json", "w", encoding="utf-8") as f:
        json.dump(request, f)

    print(f"\n[{i}/{len(questions)}] {item['id']}")
    print(item["question"])

    start = time.perf_counter()

    result = subprocess.run(
        [
            "curl.exe",
            "-sN",
            "-X",
            "POST",
            "http://127.0.0.1:8000/ask/stream",
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            "@_current_request.json",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    output = result.stdout

    meta = None
    done = None

    for line in output.splitlines():
        if not line.startswith("data: "):
            continue

        try:
            data = json.loads(line[6:])
        except json.JSONDecodeError:
            continue

        if data.get("type") == "meta":
            meta = data

        elif data.get("type") == "done":
            done = data

    row = {
        "id": item["id"],
        "elapsed_ms": round(elapsed_ms, 2),
        "cached": meta.get("cached") if meta else None,
        "cache_type": meta.get("cache_type") if meta else None,
        "ttft_ms": meta.get("ttft_ms") if meta else None,
        "cost_usd": done.get("cost_usd") if done else None,
        "latency_ms": done.get("latency_ms") if done else None,
    }

    results.append(row)

    print(row)

with open("final_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

valid = [r for r in results if r["cost_usd"] is not None]

costs = [r["cost_usd"] for r in valid]
latencies = [r["latency_ms"] for r in valid]
ttfts = [r["ttft_ms"] for r in valid if r["ttft_ms"] is not None]

print(f"Requests: {len(valid)}")
print(f"Average cost/query: ${statistics.mean(costs):.6f}")
print(f"Average latency: {statistics.mean(latencies):.2f} ms")
print(f"Average TTFT: {statistics.mean(ttfts):.2f} ms")

cached = [r for r in valid if r["cached"]]
exact = [r for r in valid if r["cache_type"] == "exact"]
semantic = [r for r in valid if r["cache_type"] == "semantic"]

print(f"Cached: {len(cached)}/{len(valid)}")
print(f"Exact cache hits: {len(exact)}")
print(f"Semantic cache hits: {len(semantic)}")

if valid:
    print(f"Cache hit rate: {len(cached) / len(valid):.2%}")

print("=" * 70)

