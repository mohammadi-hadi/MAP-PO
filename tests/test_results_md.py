"""RESULTS.md must be exactly what the committed artifacts produce."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_results_md_matches_artifacts():
    results = ROOT / "RESULTS.md"
    before = results.read_bytes()
    try:
        subprocess.run(
            [sys.executable, "src/evaluation/generate_results_md.py"],
            cwd=ROOT, check=True, capture_output=True, text=True)
        after = results.read_bytes()
    finally:
        results.write_bytes(before)
    assert after == before, (
        "RESULTS.md is stale — run src/evaluation/generate_results_md.py "
        "and commit the result")
