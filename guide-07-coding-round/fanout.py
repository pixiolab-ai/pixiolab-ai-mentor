import asyncio
import os
import anthropic

MODEL = os.getenv("LLM_MODEL", "claude-haiku-5-5")  # set to a current model ID
client = anthropic.AsyncAnthropic()

async def summarize(doc: str) -> str:
    resp = await client.messages.create(
        model=MODEL, max_tokens=400,
        messages=[{"role": "user", "content": f"Summarize:\n\n{doc}"}])
    return "".join(b.text for b in resp.content if b.type == "text")

async def map_bounded(fn, items, limit=8, timeout_s=30.0):
    """At most `limit` calls in flight; failures come back as exceptions."""
    sem = asyncio.Semaphore(limit)
    async def one(item):
        async with sem:
            return await asyncio.wait_for(fn(item), timeout_s)
    return await asyncio.gather(*(one(x) for x in items),
                                return_exceptions=True)
