"""Shared test setup for guide-02-agents. pytest loads this file first.

Placeholder API key, and a base URL on a closed local port: no test reaches
a real model. Tests that need a model replace the client with a scripted
fake (tests/fakes.py). Paths in the examples are relative to this folder.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

os.environ["ANTHROPIC_API_KEY"] = "test-key-not-real"
os.environ["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:9"
