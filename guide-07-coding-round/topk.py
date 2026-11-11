import numpy as np

def top_k_cosine(query: np.ndarray, docs: np.ndarray, k: int = 5):
    """query: shape (d,). docs: shape (n, d). Returns (ids, scores)."""
    n = docs.shape[0]
    k = min(k, n)
    if k <= 0:
        return np.array([], dtype=int), np.array([])
    q = query / (np.linalg.norm(query) + 1e-12)
    d = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-12)
    scores = d @ q                               # cosine, shape (n,)
    ids = np.argpartition(-scores, k - 1)[:k]    # O(n), unordered
    ids = ids[np.argsort(-scores[ids])]          # sort only the k winners
    return ids, scores[ids]
