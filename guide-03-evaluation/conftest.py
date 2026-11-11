"""Shared test setup for guide-03-evaluation. pytest loads this file first.

Placeholder API key and a base URL on a closed local port: no test reaches
a real model. The judge and the ticket summarizer are tested with fake
clients or the real SDK on an in-memory transport. The scripts read and
write evals/ relative to this folder, as in the guide.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

os.environ["ANTHROPIC_API_KEY"] = "test-key-not-real"
os.environ["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:9"
