"""Chapter 08: measure retrieval on docs/ with a small labeled set.

Compares four configurations from Chapter 03 on the same questions, overall
and per question category:

  dense            vector search alone
  bm25             keyword search alone
  hybrid (RRF)     both, fused with reciprocal rank fusion
  hybrid + rerank  rag.search(): RRF, then the cross-encoder

Retrieval runs locally: no API key and no model call (the first run
downloads the two small open models). With nine questions one question is
11 points, so read the numbers as a smoke test, not a benchmark.

    python run_eval.py                 # docs/ and golden.json, k = 1
    python run_eval.py --k 3
"""
import argparse
import json
import pathlib
from collections import defaultdict

import rag
from eval_retrieval import evaluate


def load_index(folder: str) -> rag.Index:
    docs = {p.stem: p.read_text()
            for p in sorted(pathlib.Path(folder).glob("*.txt"))}
    return rag.Index([c for d, t in docs.items() for c in rag.chunk_doc(d, t)])


def load_golden(path: str) -> list[dict]:
    with open(path) as f:
        rows = json.load(f)
    return [{**row, "relevant": set(row["relevant"])} for row in rows]


def retrievers(index: rag.Index, k_each: int = 50) -> dict:
    """Each one maps a question to a ranked list of chunk ids."""
    def ids(positions):
        return [index.chunks[i].id for i in positions]

    return {
        "dense": lambda q: ids(index.dense(q, k_each)),
        "bm25": lambda q: ids(index.keyword(q, k_each)),
        "hybrid (RRF)": lambda q: ids(rag.rrf(
            [index.dense(q, k_each), index.keyword(q, k_each)])),
        # The guide's line: retrieve = lambda q: [c.id for c, _ in search(...)]
        "hybrid + rerank": lambda q: [
            c.id for c, _ in rag.search(index, q, k_final=20)],
    }


def report(index: rag.Index, golden: list[dict], k: int) -> dict:
    by_category = defaultdict(list)
    for row in golden:
        by_category[row["category"]].append(row)
    results = {}
    print(f"{'configuration':<17} {'slice':<11} {f'recall@{k}':>9} "
          f"{f'hit@{k}':>6} {'MRR':>6} {'n':>3}")
    for name, retrieve in retrievers(index).items():
        slices = {"ALL": golden, **by_category}
        results[name] = {s: evaluate(retrieve, rows, k=k)
                         for s, rows in slices.items()}
        for s, m in results[name].items():
            print(f"{name:<17} {s:<11} {m[f'recall@{k}']:>9.2f} "
                  f"{m[f'hit_rate@{k}']:>6.2f} {m['mrr']:>6.2f} {m['n']:>3}")
    return results


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--docs", default="docs")
    parser.add_argument("--golden", default="golden.json")
    parser.add_argument("--k", type=int, default=1)
    args = parser.parse_args(argv)
    return report(load_index(args.docs), load_golden(args.golden), args.k)


if __name__ == "__main__":
    main()
