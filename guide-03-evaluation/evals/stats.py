import random
from math import comb

def mcnemar_exact(old, new):
    """Exact McNemar test. old, new: pass/fail booleans per case."""
    b = sum(o and not n for o, n in zip(old, new))  # got worse
    c = sum(n and not o for o, n in zip(old, new))  # got better
    m = b + c
    if m == 0:
        return b, c, 1.0
    tail = sum(comb(m, i) for i in range(min(b, c) + 1)) / 2 ** m
    return b, c, min(1.0, 2 * tail)

def paired_bootstrap(old, new, iters=10_000, seed=0):
    """Mean difference new - old with a 95% CI, resampling cases.
    old, new: per-case scores (0/1, or a mean over trials)."""
    rng = random.Random(seed)
    diffs = [b - a for a, b in zip(old, new)]
    n = len(diffs)
    means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(iters))
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters) - 1]
    return sum(diffs) / n, lo, hi

# With 3 trials per case, average per case first, then compare:
# old = [sum(t) / len(t) for t in old_trials]   # one score per case
# diff, lo, hi = paired_bootstrap(old, new)