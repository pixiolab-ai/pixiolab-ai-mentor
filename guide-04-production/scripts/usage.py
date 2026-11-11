"""Feature 02, Step 1 checkpoint: the four token counts a response reports.

Run from guide-04-production/:  python -m scripts.usage
Makes one real call (needs ANTHROPIC_API_KEY).
"""
import anthropic

from app.config import PRIMARY_MODEL

client = anthropic.Anthropic()
resp = client.messages.create(
    model=PRIMARY_MODEL, max_tokens=2048,
    output_config={"effort": "low"},
    messages=[{"role": "user", "content": "Explain RAG in one paragraph."}],
)
u = resp.usage
print("input, uncached:", u.input_tokens)
print("cache read:     ", u.cache_read_input_tokens or 0)
print("cache write:    ", u.cache_creation_input_tokens or 0)
print("output:         ", u.output_tokens)  # includes thinking tokens
