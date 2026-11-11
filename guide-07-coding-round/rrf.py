from collections import defaultdict

def rrf(rankings: list[list[str]], k: int = 60, top_n: int = 10):
    """Reciprocal rank fusion: score(d) = sum of 1 / (k + rank)."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        seen = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:          # count a doc once per list
                continue
            seen.add(doc_id)
            scores[doc_id] += 1.0 / (k + rank)
    # sort by score, then id, so ties are deterministic
    fused = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return fused[:top_n]

bm25 = ["A", "B", "C"]
dense = ["B", "D", "A"]
print(rrf([bm25, dense]))   # B 0.03252, A 0.03227, D 0.01613, C 0.01587
