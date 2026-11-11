import json, os
from harness import run_suite, summarize
from app.tickets import summarize_ticket   # the feature under test

VALID = {"billing", "account", "technical"}
GRADERS = {  # separate checks, so a failure names the check that broke
    "valid_category": lambda c, o: o["category"] in VALID,
    "right_category": lambda c, o: o["category"] == c["expect"]["category"],
    "key_facts": lambda c, o: all(
        f.lower() in o["summary"].lower() for f in c["expect"]["key_facts"]),
}

with open("evals/tickets_v3.jsonl") as f:
    cases = [json.loads(line) for line in f if line.strip()]
records = run_suite(cases, summarize_ticket, GRADERS, trials=3)
meta = {"dataset": "tickets_v3", "prompt": "summary@v4",
        "model": os.environ["LLM_MODEL"], "commit": os.getenv("GIT_SHA")}
with open("evals/results.json", "w") as f:
    json.dump({"meta": meta, "summary": summarize(records),
               "records": records}, f, indent=1)