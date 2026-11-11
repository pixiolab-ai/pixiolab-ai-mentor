"""Chapter 04, offline: cheap checks first, the critic only when they pass.

generate() returns canned drafts of a slugify function, run_checks() runs
real assertions against each draft, and critique() is a rubric stand-in. In
a real system, generate and critique are model calls (ideally a different
model or prompt for the critic) and run_checks runs your tests and linters.

Run from guide-02-agents/:  python -m examples.reflection_demo
"""
from agent.reflection import Verdict, reflect

DRAFTS = iter([
    "def slugify(s):\n    return s.lower().replace(' ', '-')",
    "import re\n\ndef slugify(s):\n"
    "    return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')",
])
CASES = {"Hello World": "hello-world", "  Rock & Roll  ": "rock-roll"}


def generate(task: str, draft: str | None, issues: list[str]) -> str:
    print(f"  generate (fixing {issues or 'nothing yet'})")
    return next(DRAFTS)


def run_checks(draft: str) -> list[str]:
    namespace: dict = {}
    # Safe here because the drafts are fixed strings in this file. Run code
    # a model wrote only inside a sandbox.
    exec(draft, namespace)
    failures = [f"slugify({given!r}) should be {want!r}"
                for given, want in CASES.items()
                if namespace["slugify"](given) != want]
    print(f"  checks: {failures or 'all passed'}")
    return failures


def critique(task: str, draft: str) -> Verdict:
    print("  critic reads the draft against the rubric")
    return Verdict(passed=True, issues=[], score=8)


def main() -> dict:
    outcome = reflect("Write slugify(s) for URL slugs", generate, critique,
                      run_checks)
    print(f"status: {outcome['status']} after {outcome.get('attempts')} "
          f"attempt(s)")
    return outcome


if __name__ == "__main__":
    main()
