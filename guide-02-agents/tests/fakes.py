"""Test helpers: a scripted stand-in for anthropic.Anthropic().messages, and
a way to import agent.state without touching this folder.

Each call to create() returns the next scripted response and records the
request, so a test can check what the loop sent back to the model.
"""
import importlib
import sys
from types import SimpleNamespace as NS


def import_state_in(directory, monkeypatch):
    """agent.state opens runs.db in the working directory when imported, so
    (re)import it inside `directory`, a temporary folder."""
    monkeypatch.chdir(directory)
    if "agent.state" in sys.modules:
        return importlib.reload(sys.modules["agent.state"])
    return importlib.import_module("agent.state")


def text(t: str) -> NS:
    return NS(type="text", text=t)


def thinking() -> NS:
    return NS(type="thinking", thinking="", signature="sig")


def tool_use(id: str, name: str, input: dict) -> NS:
    return NS(type="tool_use", id=id, name=name, input=input)


def response(*blocks, stop="end_turn", input_tokens=100, output_tokens=20):
    return NS(content=list(blocks), stop_reason=stop,
              usage=NS(input_tokens=input_tokens, output_tokens=output_tokens))


class ScriptedClient:
    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []
        self.messages = NS(create=self._create)

    def _create(self, **request):
        # Copy the history: the loop keeps appending to the same list.
        self.requests.append({**request, "messages": list(request["messages"])})
        return self.responses.pop(0)
