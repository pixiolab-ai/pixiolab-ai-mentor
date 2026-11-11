"""Feature 01, Step 5 checkpoint: watch the retry and fallback counters.

Run from guide-04-production/:  python -m scripts.metrics_server
then open http://localhost:8001/metrics in a browser. Makes five real calls.
To force fallbacks, set FALLBACK_MODEL and ANTHROPIC_BASE_URL=http://localhost:9
(nothing listens there), as in the guide.
"""
from prometheus_client import start_http_server

from app.llm.client import generate

if __name__ == "__main__":
    start_http_server(8001)  # one exporter per process
    for _ in range(5):
        r = generate([{"role": "user", "content": "Say hi in three words."}])
        print(r.model, r.fallback, r.text)
    input("Metrics are live. Press Enter to stop.")
