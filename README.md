# PixioLab AI Mentor: companion code

[![tests](https://github.com/pixiolab-ai/pixiolab-ai-mentor/actions/workflows/tests.yml/badge.svg)](https://github.com/pixiolab-ai/pixiolab-ai-mentor/actions/workflows/tests.yml)

Runnable, tested code for the **PixioLab AI Mentor** series: ten guides by PixioLab for software engineers moving into AI engineering. The guides teach the ideas and print the important code; this repository holds that code in full, with tests, so you can run it, break it and watch how each system works.

Every folder is a small, self-contained project with its own README. Each README explains which chapters it accompanies, how the system is put together, how one request flows through the code, why it is built that way, and how to run and test it.

## The series

| No | Guide | What it covers |
|---|---|---|
| 00 | The 45-Day AI Engineer Roadmap | Start here. The week-by-week plan that ties every other guide together. |
| 01 | RAG, Explained | Retrieval-augmented generation from first principles to production. |
| 02 | Agentic Design Patterns | How to design, build, evaluate and deploy agents. |
| 03 | Evaluation Field Guide | Proving AI systems work, and keeping them working. |
| 04 | Production AI Features | Six features that turn a demo into a product, with full code. |
| 05 | The Portfolio Project Catalog | 17 portfolio projects in three tiers, with specs, ship gates and interview stories. |
| 06 | System Design Interviews | Running the room, core building blocks, AI system design. |
| 07 | AI Engineering Interview Q&A | An 80+ question bank with strong answers, follow-ups and level tags. |
| 08 | When You Don't Know the Answer | A playbook for answering questions outside your experience. |
| 09 | Get the Offer | From finished project to signed offer: targeting, resume, referrals, take-homes and negotiation. |

The guides are PDFs distributed by PixioLab. They are not part of this repository.

## Guide to folder

| Guide | Folder | What's inside | Tests |
|---|---|---|---|
| 01 · RAG, Explained | [`guide-01-rag/`](guide-01-rag/) | A minimal RAG pipeline: chunking, embeddings, BM25 plus vector search, reciprocal rank fusion, reranking, cited generation; retrieval metrics and an eval runner | 17 |
| 02 · Agentic Design Patterns | [`guide-02-agents/`](guide-02-agents/) | Pattern skeletons: the tool-use loop, a tool registry, planning, reflection, routing, multi-agent, budgets, checkpoints, an MCP server, trajectory checks | 41 |
| 03 · Evaluation Field Guide | [`guide-03-evaluation/`](guide-03-evaluation/) | A tiny eval harness, an LLM judge, judge calibration (TPR/TNR, kappa), paired statistics, retrieval and agent metrics, a ship gate | 34 |
| 04 · Production AI Features | [`guide-04-production/`](guide-04-production/) | The full production app: retries and fallback, cost tracking, prompt regression tests, scheduled jobs, a request queue, permission-aware RAG; Docker Compose | 77 |
| 07 · AI Engineering Interview Q&A | [`guide-07-coding-round/`](guide-07-coding-round/) | The eight coding-round solutions from Module 09 | 31 |

Guides 00, 05, 06, 08 and 09 have no folder. Guide 00 is the plan that sends you to the others. Guide 05's portfolio projects are yours to build: each folder's exercises point to the projects that grow out of it. Guides 06, 08 and 09 are about interviews and the job search, not code.

**Building the 45-day capstone?** Don't start here: create your repo from the [sourcebook-starter](https://github.com/pixiolab-ai/sourcebook-starter) template. It has the capstone's layout and a test workflow with none of the work done. Come back here to check your build against a reference.

## Five-minute quickstart

You need Python 3.11 or later and Git. No API key is needed for this part.

```bash
git clone https://github.com/pixiolab-ai/pixiolab-ai-mentor.git
cd pixiolab-ai-mentor/guide-02-agents
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt

python -m pytest                     # 41 tests, all offline
python -m examples.planning_demo     # a plan fails, gets revised, and finishes
python -m examples.mcp_client        # an MCP client calls the orders server
```

Then add a key and run a real agent:

```bash
cp .env.example .env                 # put your ANTHROPIC_API_KEY in it
set -a; . ./.env; set +a
python -m examples.support_agent "Where is my order ORD-12345678?"
```

Every folder works the same way: `cd` into it, install `requirements.txt` to run the code (or `requirements-dev.txt` to run the tests), and run commands from inside the folder.

## Suggested learning path

Follow Guide 00's roadmap. It points you to one guide per week, and this table adds the folder to open alongside it.

| Roadmap week (Guide 00) | Read | Run |
|---|---|---|
| Weeks 1 and 2 · How LLMs work; prompts, schemas and tools | Guide 00, then Guide 04 Chapter 00 and Feature 01 for calls that survive errors | `guide-04-production`: `scripts/block_types.py`, `app/llm/`, `tests/test_llm_client.py` |
| Week 3 · Retrieval and RAG | Guide 01 | `guide-01-rag` |
| Week 4 · Agents and tools | Guide 02 | `guide-02-agents` |
| Week 5 · Evaluation and production | Guide 03, then Guide 04 Features 02 and 03 | `guide-03-evaluation`, then `guide-04-production` |
| Week 6 · Safety, ML foundations and the ship gate | Guide 04 Feature 06, and Guide 05 · P02 Tripwire | `guide-04-production` (Features 03 and 06) |
| Interview preparation | Guides 06, 07 and 08 | `guide-07-coding-round` |

The roadmap's capstone is one repository you build yourself, Guide 05 · P04 Sourcebook with P01 Groundtruth (and P02 Tripwire if there's time). Use these folders as references while you build it, not as the project.

## Using this repository with the guides

- **The printed code is here, line for line.** Where a guide prints an excerpt or splits a file across several listings, the repository has the complete file. Blank lines between definitions can differ from the PDF.
- **Each folder README lists its differences from the PDF**, such as a file that the guide shows in two states as a feature grows, or a version pin changed so the code installs on Python 3.11.
- **Additions are labeled.** Examples, eval runners, sample data and tests that the guides don't print are marked as additions in the folder READMEs.

## Conventions

- **Python 3.11 or later.** CI runs every folder on Python 3.11 and 3.13.
- **Model IDs come from environment variables**, with current Anthropic IDs as defaults (for example `LLM_MODEL=claude-sonnet-5-5`). Model IDs go stale: set them to current ones from your provider's documentation. Prices in the code are illustrative; copy real ones from your provider's pricing page.
- **No secrets in code.** Each folder has a `.env.example` with placeholders. Copy it to `.env`, which Git ignores.
- **Tests mock the model APIs.** No test calls a model, so the suite runs without keys, in seconds, and for free. Each folder's README explains how its tests mock the APIs and how to run the code against the real ones.
- **Guide 04's database tests run against Postgres with pgvector.** Locally they are skipped unless you set `DATABASE_URL`; CI provides a Postgres service container.

## Repository layout

```text
pixiolab-ai-mentor/
├── README.md                  this file
├── LICENSE                    MIT
├── .github/workflows/tests.yml  runs every folder's tests with mocked model APIs
├── guide-01-rag/              rag.py, eval_retrieval.py, run_eval.py, search_tool.py, docs/
├── guide-02-agents/           agent/, servers/, evals/, tools/, examples/
├── guide-03-evaluation/       evals/, app/tickets.py
├── guide-04-production/       app/, db/, evals/, prompts/, ops/, scripts/, docker-compose.yml
└── guide-07-coding-round/     retry.py, bucket.py, chunker.py, topk.py, rrf.py, toolstream.py, evalh.py, fanout.py
```

Each folder also has `tests/`, `requirements.txt`, `requirements-dev.txt`, `.env.example` and its own README.

## Run every test

```bash
for d in guide-01-rag guide-02-agents guide-03-evaluation guide-04-production guide-07-coding-round; do
  (cd "$d" && python -m pytest -q) || break
done
```

Use a separate virtual environment per folder, or one with every folder's `requirements-dev.txt` installed. Run each folder on its own: two folders have an `app` package, so one pytest run across the whole repository would mix them up.

## Reporting a problem

If the code here and a guide disagree, or something doesn't run as described, [open an issue](https://github.com/pixiolab-ai/pixiolab-ai-mentor/issues) with the guide number, the chapter and what you saw.

## License

MIT. Copyright (c) 2026 PixioLab. See [LICENSE](LICENSE).
