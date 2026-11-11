def recall_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & relevant) / len(relevant)

def hit_at_k(retrieved, relevant, k):
    return float(any(r in relevant for r in retrieved[:k]))

def reciprocal_rank(retrieved, relevant):
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0

def evaluate(retrieve, golden, k=5):
    """golden: [{"question": str, "relevant": {chunk ids}}, ...]"""
    rows = []
    for g in golden:
        got = retrieve(g["question"])          # ranked list of chunk ids
        rel = g["relevant"]
        rows.append((recall_at_k(got, rel, k), hit_at_k(got, rel, k),
                     reciprocal_rank(got, rel)))
    n = len(rows)
    return {f"recall@{k}": sum(r[0] for r in rows) / n,
            f"hit_rate@{k}": sum(r[1] for r in rows) / n,
            "mrr": sum(r[2] for r in rows) / n, "n": n}
