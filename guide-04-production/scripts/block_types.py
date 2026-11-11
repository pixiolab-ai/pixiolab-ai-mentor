"""Chapter 00 checkpoint: see why you read blocks by type, never by position.

Run from guide-04-production/:  python -m scripts.block_types
Makes one real call (needs ANTHROPIC_API_KEY).
"""
import anthropic

from app.config import PRIMARY_MODEL
from app.llm.text import claude_text

resp = anthropic.Anthropic().messages.create(
    model=PRIMARY_MODEL, max_tokens=2048,
    messages=[{"role": "user", "content": "Name three prime numbers."}])
print([block.type for block in resp.content])  # may start with "thinking"
print(claude_text(resp))
