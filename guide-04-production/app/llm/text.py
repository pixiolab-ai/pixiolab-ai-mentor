"""Read a model's answer safely. Every feature in this guide uses it."""


class NoText(Exception):
    """The model gave no usable answer. The same request would fail the
    same way, so this error is never retried."""

    def __init__(self, reason: str, result=None):
        super().__init__(reason)
        self.reason = reason  # "refusal" | "max_tokens" | "empty"
        self.result = result  # token usage, so the call can still be billed


def claude_text(resp) -> str:
    """Join the text blocks of an Anthropic Message.

    Never read resp.content[0].text: current models think by default, so
    the first block can be a thinking block, and a refusal can come back
    with no text at all.
    """
    if resp.stop_reason in ("refusal", "max_tokens"):
        raise NoText(resp.stop_reason)  # declined, or cut off mid-answer
    text = "".join(b.text for b in resp.content if b.type == "text")
    if not text.strip():
        raise NoText("empty")
    return text
