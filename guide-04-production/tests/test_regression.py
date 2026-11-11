import json
from pathlib import Path

# Tolerances are counted in cases, not percent. With 40 cases, one flip
# moves the pass rate by 2.5 points, so a 2% limit would fail on noise.
MAX_NET_REGRESSIONS = 1    # cases that broke minus cases that got fixed
MAX_JUDGE_DROP = 0.25      # average judge score, 1 to 5 scale


def load(name: str) -> dict:
    return json.loads(Path("evals", name).read_text())


def test_no_regression():
    base, new = load("baseline.json"), load("results.json")
    old, cur = base["cases"], new["cases"]
    broke = [c for c in cur if c in old
             and old[c]["passed"] and not cur[c]["passed"]]
    fixed = [c for c in cur if c in old
             and not old[c]["passed"] and cur[c]["passed"]]
    for cid in broke:  # name the cases, not just the average
        print(f"REGRESSED {cid}: {cur[cid]['trials'][0]}")
    critical = [c for c in cur if cur[c]["critical"] and not cur[c]["passed"]]
    assert not critical, f"critical cases failed: {critical}"
    assert len(broke) - len(fixed) <= MAX_NET_REGRESSIONS, (
        f"{len(broke)} cases broke, {len(fixed)} fixed")
    if base["summary"]["avg_judge"] and new["summary"]["avg_judge"]:
        drop = base["summary"]["avg_judge"] - new["summary"]["avg_judge"]
        assert drop <= MAX_JUDGE_DROP, f"judge average fell by {drop:.2f}"
