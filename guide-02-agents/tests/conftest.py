import pytest

import agent.tools as tools


@pytest.fixture(autouse=True)
def restore_tool_registry():
    """The examples register backends in the shared TOOLS registry; put it
    back after every test so tests don't depend on their order."""
    saved = {name: dict(spec) for name, spec in tools.TOOLS.items()}
    yield
    tools.TOOLS.clear()
    tools.TOOLS.update(saved)
