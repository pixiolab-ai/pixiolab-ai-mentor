"""Shared test setup for guide-07-coding-round. pytest loads this file first.

fanout.py creates an Anthropic client at import time. The tests never call
it (they pass their own async function to map_bounded); the placeholder key
and closed local port make sure nothing could reach a real model.
"""
import os

os.environ["ANTHROPIC_API_KEY"] = "test-key-not-real"
os.environ["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:9"
