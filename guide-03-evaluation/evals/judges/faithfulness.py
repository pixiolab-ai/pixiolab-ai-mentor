import json, os
import anthropic

# Set to a current model ID. Prefer a different model or family from the
# one that generated the answers, to limit self-preference bias.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-opus-5-5")
client = anthropic.Anthropic()

RUBRIC = """You check one thing: is every factual claim in the ANSWER
supported by the CONTEXT?
PASS: every claim is stated in, or directly follows from, the CONTEXT.
Saying that the CONTEXT does not contain the answer is a PASS.
FAIL: at least one claim is absent from, or contradicts, the CONTEXT.
Ignore style, tone and length. List each claim with a quote from the
CONTEXT that supports it (or "none"), then give the verdict."""

CLAIM = {"type": "object", "additionalProperties": False,
         "required": ["claim", "evidence", "supported"],
         "properties": {"claim": {"type": "string"},
                        "evidence": {"type": "string"},
                        "supported": {"type": "boolean"}}}
VERDICT = {"type": "string", "enum": ["PASS", "FAIL"]}
SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["claims", "verdict"],
          "properties": {"claims": {"type": "array", "items": CLAIM},
                         "verdict": VERDICT}}

def judge_faithfulness(context: str, answer: str) -> dict:
    resp = client.messages.create(
        model=JUDGE_MODEL, max_tokens=16000, system=RUBRIC,
        messages=[{"role": "user", "content":
                   f"<context>\n{context}\n</context>\n"
                   f"<answer>\n{answer}\n</answer>"}],
        output_config={"format": {"type": "json_schema",
                                  "schema": SCHEMA}})
    text = "".join(b.text for b in resp.content if b.type == "text")
    if resp.stop_reason != "end_turn" or not text:
        return {"verdict": "ERROR", "stop_reason": resp.stop_reason}
    return json.loads(text)