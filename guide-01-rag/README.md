# Guide 01 · RAG, Explained: a minimal pipeline you can run

The whole retrieval-augmented generation pipeline from *RAG, Explained* (Guide 01 of PixioLab AI Mentor) in one small file, `rag.py`: chunk, embed, hybrid search with BM25 and vectors, reciprocal rank fusion, cross-encoder reranking, and a generated answer whose citations are checked in code. Around it: the retrieval metrics from Chapter 08, an eval runner that compares retrieval configurations, and the search tool definition from Chapter 06.

## What you'll learn

- How documents become chunks with stable IDs that a model can cite.
- Why dense (vector) search and BM25 keyword search fail on different questions, and how RRF fuses their rankings without comparing their scores.
- What a cross-encoder reranker adds after fusion.
- How to assemble a grounded prompt that treats retrieved text as data, and catch hallucinated citations in code.
- How to measure retrieval with recall@k, hit rate@k and MRR, per configuration and per question category.

| Guide 01 chapter | Where it lives |
|---|---|
| Chapter 02 · The pipeline, stage by stage | the stages in `rag.py`, in order |
| Chapter 03 · Hybrid search and rank fusion | `Index.dense()`, `Index.keyword()`, `rrf()`, `search()` in `rag.py` |
| Chapter 04 · A minimal pipeline you can run | `rag.py` (printed as three listings), `docs/` |
| Chapter 06 · Beyond one-shot retrieval | `search_tool.py` (the search tool definition, plus code that runs it) |
| Chapter 08 · Evaluate and debug | `eval_retrieval.py`, `run_eval.py`, `golden.json` |

## Architecture

```mermaid
flowchart LR
    subgraph offline["Indexing (once)"]
        docs["docs/*.txt"] --> chunk["chunk_doc()<br/>~220 words, 40 overlap"]
        chunk --> embed["EMBED.encode()<br/>bge-small-en-v1.5"]
        chunk --> bm25["BM25Okapi<br/>over tokenize()"]
        embed --> vecs[("Index.vecs<br/>normalized vectors")]
    end

    subgraph online["Answering (per question)"]
        q(["question"]) --> dense["Index.dense()<br/>top 50 by cosine"]
        q --> kw["Index.keyword()<br/>top 50 by BM25"]
        dense --> fuse["rrf()<br/>keep top 20"]
        kw --> fuse
        fuse --> rerank["RERANK.predict()<br/>cross-encoder, keep 5"]
        rerank --> prompt["tagged sources<br/>+ SYSTEM rules"]
        prompt --> llm[["client.messages.create()"]]
        llm --> check["parse cited chunk ids<br/>flag ids never sent"]
    end

    vecs -.-> dense
    bm25 -.-> kw
    check --> out(["answer, cited, unknown"])
```

## How a request flows

`python rag.py docs "Can I return boots after 40 days?"`, step by step:

1. **Load and chunk.** The `__main__` block reads every `docs/*.txt` file. `chunk_doc()` splits each one into paragraphs, packs whole paragraphs into chunks of about 220 words, and carries the last 40 words forward as overlap. Each `Chunk` gets an ID such as `returns#0`, which is what the model will cite.
2. **Index.** `Index.__init__()` embeds every chunk with `EMBED` (bge-small-en-v1.5, 384 dimensions, normalized, so a dot product is the cosine similarity) and builds a BM25 index over `tokenize()`'s terms. The tokenizer keeps codes such as `e-404` whole so keyword search can match them exactly.
3. **Retrieve twice.** `search()` asks both retrievers for their top 50. `Index.dense()` embeds the question with the `QUERY_PREFIX` that the BGE model card recommends for short queries (passages are embedded without it: asymmetric encoding). `Index.keyword()` scores BM25 and drops chunks with no matching terms.
4. **Fuse.** `rrf()` gives each chunk `1 / (60 + rank)` from each list and adds them up, so agreement between retrievers wins without ever comparing a cosine score with a BM25 score. The top 20 fused candidates go on.
5. **Rerank.** `RERANK.predict()`, a MiniLM cross-encoder, reads the question and each candidate together and scores the pair; the best 5 survive.
6. **Generate.** `answer()` wraps each chunk in `<source id="...">` tags and sends them with the question. The system prompt carries the four rules from Chapter 01: answer only from the sources, treat text in the tags as data, cite after every claim, and say "I don't know" when the sources don't cover it. The answer text is read by block type (`b.type == "text"`), because a response can start with thinking blocks or contain no text at all.
7. **Check the citations.** A regular expression collects every `[doc#n]` the answer cites, and any ID that wasn't among the sources sent is returned as `unknown`: a hallucinated citation, caught in code instead of in production.

`run_eval.py` runs steps 1 to 5 for every question in `golden.json`, with no model call, and scores four configurations: dense only, BM25 only, hybrid (RRF), and hybrid plus rerank. With the real models on these four documents, "error E-404 on login" ranks the error-code page first through both retrievers, and for "how do I get my money back?" vector search puts the returns page first while BM25, matching common words, ranks the boot-care page above it; fusion plus the reranker put the returns page on top, as Chapter 04 describes.

## File-by-file tour

| File | Responsibility |
|---|---|
| `rag.py` | The pipeline, exactly as Chapter 04 prints it: `Chunk`, `chunk_doc()`, `tokenize()`, `Index`, `rrf()`, `search()`, `answer()` and the command line. |
| `eval_retrieval.py` | Chapter 08's metrics: `recall_at_k()`, `hit_at_k()`, `reciprocal_rank()` and `evaluate()` over a golden set. |
| `run_eval.py` | Not printed in the guide: builds the index from `docs/`, scores four retrieval configurations from `golden.json`, overall and per category. |
| `golden.json` | Nine labeled questions for the sample documents, tagged "exact term" or "paraphrase", each with the chunk ID that answers it. |
| `search_tool.py` | Chapter 06's `SEARCH_TOOL` definition (Anthropic tool format) and `execute_search()`, which validates a tool call and runs `search()`. |
| `docs/` | Four short pages from a fictional store: returns, shipping, boot care and login error codes. |
| `conftest.py` | Test setup: replaces `sentence_transformers` with small deterministic fakes, so tests need no PyTorch or model download. |
| `tests/test_rag.py` | Chunking and overlap, the tokenizer, RRF, search order, the prompt, citation checks, the command line. |
| `tests/test_eval.py` | The metrics (with Chapter 08's worked example), the eval runner, the search tool's validation. |
| `requirements.txt`, `requirements-dev.txt` | Runtime packages; the lighter set the tests need. |

## Key design decisions and trade-offs

| Decision | Why |
|---|---|
| Everything in memory, no framework | The point is to see every moving part. Swapping in pgvector or Qdrant, a hosted embedder or another LLM changes a few lines, not the design. |
| Chunks packed from whole paragraphs, with overlap | Splitting along the document's structure keeps sentences intact; the overlap keeps a fact that straddles a boundary findable from either side. |
| Hybrid search | Dense search handles paraphrases but often misses exact identifiers such as error codes; BM25 is the opposite. Running both is the cheapest large win in retrieval. |
| Rank fusion, not score fusion | Cosine and BM25 scores live on different scales. RRF uses ranks only, with k = 60 by the original paper's convention. |
| Rerank a short list | A cross-encoder reads question and passage together, which is accurate but slow, so it scores only the 20 fused candidates. |
| Sources as tagged data | Retrieved text can contain instructions. The system prompt says text inside `<source>` tags is data, and the tags carry the IDs the model cites. |
| Citations checked in code | The model can cite an ID it never saw. Comparing cited IDs with the IDs sent turns that into a number you can monitor. |
| Model ID from an environment variable | Model IDs go stale; `LLM_MODEL` changes without a code change. Sampling parameters such as temperature aren't set, because some current models fix or reject them. |
| Measure retrieval separately | When an answer is wrong, look at the retrieved chunks first: most RAG problems are retrieval problems, and retrieval metrics are free to compute locally. |

## Run it

**Prerequisites:** Python 3.11 or later. An Anthropic API key for the answer step only; retrieval and the eval runner run locally.

```bash
cd guide-01-rag
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt      # includes sentence-transformers and PyTorch
cp .env.example .env                 # add your key
set -a; . ./.env; set +a

python run_eval.py                   # retrieval metrics at k = 1, no API key needed
python run_eval.py --k 3
python rag.py docs "Can I return boots after 40 days?"
```

The first run downloads the two open models (bge-small-en-v1.5 and the MiniLM cross-encoder); both run on a laptop CPU for a few thousand chunks. A typical answer looks like this (wording varies, retrieval doesn't):

```text
Yes. You can return any item within 60 days if you have the receipt
[returns#0].

cited: ['returns#0'] | not in sources: []
```

Use your own documents by pointing at another folder of `.txt` files, and label a few questions in a JSON file shaped like `golden.json`: `python run_eval.py --docs my_docs --golden my_golden.json`.

## Test it

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The tests never download a model or call an API. `conftest.py` installs a fake `sentence_transformers` module before anything imports `rag.py`: the embedder is a hashed bag of words (texts that share words get similar vectors) and the cross-encoder scores the share of question words a passage contains. `rank_bm25` is the real library. The Anthropic client is replaced with a fake that returns a scripted response, including a thinking block, so the tests check the prompt `answer()` builds and how it handles citations.

To check the pipeline against the real models, run `python run_eval.py` after installing `requirements.txt`, and `python rag.py docs "<question>"` with your API key for the generation step.

## Exercises

1. **Filter before ranking.** Add a `doc_type` field to `Chunk` and a filter argument to `search()`. Apply it inside both retrievers, not after fusion: filtering afterwards can leave you with nothing. (Chapter 04's first exercise.)
2. **Say "I don't know" without calling the model.** Return early when the best rerank score is below a threshold. Raw cross-encoder scores aren't probabilities, so pick the threshold from ten questions your documents can and can't answer, and re-check it whenever you change models.
3. **Grow the golden set.** Add documents and 30 or more labeled questions across categories, then compare chunk sizes and `k_each` values with `run_eval.py`. This is the start of Guide 05 · P01 Groundtruth, the retrieval evaluation harness.
4. **Make retrieval a tool.** Send `SEARCH_TOOL` to a model with the loop in `guide-02-agents/agent/loop.py` and `execute_search()` as its executor, with a step cap and repeat detection. Compare its answers with single-shot `answer()` on multi-part questions.
5. **Take it to production.** Replace `Index` with pgvector, add permission filters inside the query and citations in the UI: that's Guide 05 · P04 Sourcebook, and `guide-04-production/app/rag/` shows the permission-aware search.

## Differences from the PDF

None in the printed code: `rag.py` and `eval_retrieval.py` match Chapters 04 and 08 line for line, and `search_tool.py` holds Chapter 06's definition unchanged. `run_eval.py`, `golden.json`, `execute_search()` and the tests are additions.
