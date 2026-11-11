from math import log2

def recall_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & relevant) / len(relevant)

def precision_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & relevant) / k

def hit_at_k(retrieved, relevant, k):
    return float(any(doc in relevant for doc in retrieved[:k]))

def reciprocal_rank(retrieved, relevant):
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved, grades, k):
    """grades: {doc_id: relevance 0-3}; unlisted docs count as 0."""
    dcg = sum(grades.get(doc, 0) / log2(i + 2)
              for i, doc in enumerate(retrieved[:k]))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(g / log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0

# retrieved: ranked list of IDs; relevant: set of IDs for the query.
# Average each metric over queries, and report it per question type.