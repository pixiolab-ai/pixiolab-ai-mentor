"""Chapter 06: retrieval as a tool, for agentic RAG.

SEARCH_TOOL is the definition printed in the guide, in Anthropic's tool
format: its description is part of your retrieval quality. execute_search()
is what runs when a model calls the tool: it checks the arguments the way
the schema says, calls rag.search(), and returns errors as results the model
can act on. Guide 02 builds the loop that sends this tool to a model; plug
execute_search into its execute() function.
"""
from rag import search

SEARCH_TOOL = {
    "name": "search_docs",
    "description": ("Hybrid search over the internal help center. Returns "
                    "up to k chunks with ids, titles and updated dates. Use "
                    "specific terms; if results are off-topic, search again "
                    "with a different query instead of guessing."),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "search terms"},
            "k": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "required": ["query"],
    },
}


def execute_search(index, args: dict) -> dict:
    """Run one search_docs call against an rag.Index."""
    query, k = args.get("query"), args.get("k", 5)
    if not isinstance(query, str) or not query.strip():
        return {"ok": False, "error": "query must be a non-empty string"}
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 20:
        return {"ok": False, "error": "k must be an integer from 1 to 20"}
    hits = search(index, query, k_final=k)
    # This sample corpus has no page dates; a real help center would return
    # each page's title and last-updated date from its metadata.
    return {"ok": True, "data": [
        {"id": chunk.id, "title": chunk.doc, "updated": None,
         "text": chunk.text, "score": round(score, 3)}
        for chunk, score in hits]}
