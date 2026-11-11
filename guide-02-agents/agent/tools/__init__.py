from pydantic import BaseModel, Field, ValidationError

class SearchDocsInput(BaseModel):
    query: str = Field(description="Keywords or a short question")
    max_results: int = Field(5, ge=1, le=10)

def search_docs(query: str, max_results: int) -> list[dict]:
    raise NotImplementedError("call your search backend here")

TOOLS = {
    "search_docs": {
        "description": "Search the product docs for how-to questions. "
                       "Returns title, url and snippet for each hit.",
        "schema": SearchDocsInput,
        "fn": search_docs,
    },
}

def tool_definitions() -> list[dict]:
    return [{"name": name, "description": spec["description"],
             "input_schema": spec["schema"].model_json_schema()}
            for name, spec in TOOLS.items()]

def execute(name: str, args: dict) -> dict:
    spec = TOOLS.get(name)
    if spec is None:
        return {"ok": False, "error": f"Unknown tool {name!r}. "
                                      f"Available: {', '.join(TOOLS)}"}
    try:
        parsed = spec["schema"].model_validate(args)     # validate first
        return {"ok": True, "data": spec["fn"](**parsed.model_dump())}
    except ValidationError as e:
        return {"ok": False, "error": f"Invalid arguments: {e}"}
    except Exception as e:                               # the tool failed
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
