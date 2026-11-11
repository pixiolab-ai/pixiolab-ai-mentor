from math import sqrt

def calibration(human, judge):
    """human, judge: booleans (True = PASS) for the same items."""
    n = len(human)
    tp = sum(h and j for h, j in zip(human, judge))
    tn = sum(not h and not j for h, j in zip(human, judge))
    pos = sum(human)
    p_o = (tp + tn) / n                          # raw agreement
    p_h, p_j = pos / n, sum(judge) / n
    p_e = p_h * p_j + (1 - p_h) * (1 - p_j)      # chance agreement
    return {"tpr": tp / pos, "tnr": tn / (n - pos), "agreement": p_o,
            "kappa": (p_o - p_e) / (1 - p_e)}

def wilson(k, n, z=1.96):
    """95% Wilson interval for k successes in n trials."""
    p = k / n
    mid = p + z * z / (2 * n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (mid - half) / (1 + z * z / n), (mid + half) / (1 + z * z / n)

def corrected_pass_rate(observed, tpr, tnr):
    """Rogan-Gladen estimate of the true pass rate."""
    if tpr + tnr <= 1:
        raise ValueError("judge is no better than chance")
    return min(1.0, max(0.0, (observed + tnr - 1) / (tpr + tnr - 1)))

# Worked example from this chapter:
human = [True] * 160 + [False] * 40
judge = [True] * 150 + [False] * 10 + [False] * 26 + [True] * 14
stats = calibration(human, judge)  # tpr .938, tnr .65, kappa .61
print(wilson(26, 40))              # (0.495, 0.779)
print(corrected_pass_rate(0.78, stats["tpr"], stats["tnr"]))  # 0.732